import torch
import torch.nn as nn
from graphviz import Digraph
import os
import sys

# Add project root to path (scripts/../)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Try imports
YOLOVideo = None
YOLOMemoryAttention = None
YOLOMultiview = None
MultiviewFusionBlock = None

try:
    from ultralytics.models.yolo.video.model import YOLOVideo
    from ultralytics.nn.modules.video_block import YOLOMemoryAttention
    print("Imported YOLOVideo and YOLOMemoryAttention successfully.")
except ImportError as e:
    print(f"Could not import YOLOVideo/YOLOMemoryAttention: {e}")

try:
    from ultralytics.models.yolo.multiview.model import YOLOMultiview
    from ultralytics.nn.modules.multiview_block import MultiviewFusionBlock
    print("Imported YOLOMultiview and MultiviewFusionBlock successfully.")
except ImportError as e:
    print(f"Could not import YOLOMultiview/MultiviewFusionBlock: {e}")

if not YOLOVideo and not YOLOMultiview:
    print("Trying generic YOLO.")
    from ultralytics import YOLO

def get_layer_params(layer):
    params = []
    if isinstance(layer, nn.Conv2d):
        params.append(f"k={layer.kernel_size}")
        params.append(f"s={layer.stride}")
        params.append(f"c={layer.out_channels}")
    elif hasattr(layer, 'conv') and isinstance(layer.conv, nn.Conv2d):
        params.append(f"k={layer.conv.kernel_size}")
        params.append(f"s={layer.conv.stride}")
        params.append(f"c={layer.conv.out_channels}")
    elif hasattr(layer, 'cv1') and hasattr(layer, 'cv2'): # C2f, C3, etc.
        if hasattr(layer.cv2, 'conv'):
             params.append(f"c_out={layer.cv2.conv.out_channels}")
    
    # Memory Attention specific
    if YOLOMemoryAttention and isinstance(layer, YOLOMemoryAttention):
         params.append(f"d_model={layer.d_model}")
         params.append(f"max_mem={layer.max_memory}")

    # Multiview Fusion specific
    if MultiviewFusionBlock and isinstance(layer, MultiviewFusionBlock):
        params.append(f"hidden_dim={layer.hidden_dim}")
        params.append(f"c1={layer.c1}")
        params.append(f"c2={layer.c2}")

    return ", ".join(params)

def main():
    # Use absolute path for config file relative to project root
    # Default to multiview for this request
    model_cfg = os.path.join(project_root, 'ultralytics/cfg/models/12/yolo12-multiview.yaml')
    output_file = os.path.join(current_dir, 'yolo12_multiview_architecture')
    
    print(f"Loading model from {model_cfg}...")
    
    model_wrapper = None
    if YOLOMultiview:
        try:
            model_wrapper = YOLOMultiview(model_cfg, verbose=False)
        except Exception as e:
             print(f"Error init YOLOMultiview: {e}")
    
    if model_wrapper is None and YOLOVideo:
        try:
            model_wrapper = YOLOVideo(model_cfg, verbose=False)
        except Exception as e:
             print(f"Error init YOLOVideo: {e}")

    if model_wrapper is None:
        from ultralytics import YOLO
        model_wrapper = YOLO(model_cfg)
        
    if hasattr(model_wrapper, 'model'):
        model = model_wrapper.model
    else:
        model = model_wrapper

    # Hook to capture shapes
    layer_shapes = {}
    
    def get_shape_hook(idx):
        def hook(module, input, output):
            # input is tuple
            if input and isinstance(input[0], torch.Tensor):
                in_shape = str(tuple(input[0].shape))
            else:
                in_shape = "Complex"
                
            if isinstance(output, (list, tuple)):
                # Handle list of tensors (e.g. from Detect head)
                shapes = []
                for o in output:
                    if isinstance(o, torch.Tensor):
                        shapes.append(tuple(o.shape))
                    else:
                        shapes.append("Non-Tensor")
                out_shape = str(shapes)
            elif isinstance(output, torch.Tensor):
                out_shape = str(tuple(output.shape))
            else:
                out_shape = "Non-Tensor"
                
            layer_shapes[idx] = {'in': in_shape, 'out': out_shape}
        return hook

    # Register hooks
    if isinstance(model, nn.Sequential):
        layers = list(model.children())
    elif hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        layers = list(model.model.children())
    else:
        print(f"Unexpected model structure. Type: {type(model)}")
        return

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_shape_hook(i)))
        
    # Dummy forward pass
    print("Running dummy forward pass to capture shapes...")
    # (B, V, C, H, W) for Multiview model
    # B=1, V=2
    dummy_input = torch.randn(1, 2, 3, 640, 640) 
    dummy_coords = torch.randn(1, 2, 640, 640, 3)
    
    try:
        # Multiview forward expects coords
        if YOLOMultiview and isinstance(model_wrapper, YOLOMultiview):
             model_wrapper(dummy_input, coords=dummy_coords)
        else:
             model_wrapper(dummy_input)
    except Exception as e:
        print(f"Forward pass with 5D input failed: {e}")
        print("Trying 4D input...")
        dummy_input = torch.randn(1, 3, 640, 640)
        try:
            model_wrapper(dummy_input)
        except Exception as e2:
            print(f"Forward pass with 4D input failed: {e2}")

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Generate Graph
    print("Generating graph...")
    dot = Digraph(comment='YOLO12 Architecture', format='png')
    dot.attr(rankdir='TB') # Top to Bottom
    dot.attr(compound='true') # Allow edges between clusters
    
    # Input Node
    dot.node("input", "Input\n(1, 2, 3, 640, 640)", shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Store (input_node_id, output_node_id) for each layer index
    # -1 maps to "input"
    layer_io_nodes = {-1: ("input", "input")}

    # Layer Nodes
    for i, layer in enumerate(layers):
        layer_name = layer.__class__.__name__
        idx = getattr(layer, 'i', i)
        shapes = layer_shapes.get(i, {'in': '?', 'out': '?'})
        params = get_layer_params(layer)
        
        is_memory_attn = YOLOMemoryAttention and isinstance(layer, YOLOMemoryAttention)
        is_multiview_fusion = MultiviewFusionBlock and isinstance(layer, MultiviewFusionBlock)
        
        if is_memory_attn:
            # Create a subgraph for Memory Attention
            with dot.subgraph(name=f'cluster_{idx}') as c:
                c.attr(label=f'Layer {idx}: {layer_name}\n{params}\nIn: {shapes["in"]}\nOut: {shapes["out"]}')
                c.attr(style='filled', color='lightgrey')
                
                # Internal nodes
                in_node = f"{idx}_in"
                proj_in_node = f"{idx}_proj_in"
                mem_enc_node = f"{idx}_mem_enc"
                attn_node = f"{idx}_attn"
                proj_out_node = f"{idx}_proj_out"
                add_node = f"{idx}_add"
                
                c.node(in_node, "Split Input", shape='point')
                c.node(proj_in_node, "Proj In", shape='box', style='filled', fillcolor='white')
                c.node(mem_enc_node, "Memory\nEncoder", shape='box', style='filled', fillcolor='lightyellow')
                c.node(attn_node, "Attention\n(Core)", shape='diamond', style='filled', fillcolor='orange')
                c.node(proj_out_node, "Proj Out", shape='box', style='filled', fillcolor='white')
                c.node(add_node, "Add\n(Residual)", shape='circle', style='filled', fillcolor='white')
                
                # Internal Edges
                c.edge(in_node, proj_in_node)
                c.edge(in_node, mem_enc_node)
                c.edge(in_node, add_node, style='dashed', label='skip')
                
                c.edge(proj_in_node, attn_node, label='curr')
                c.edge(mem_enc_node, attn_node, label='memory')
                
                c.edge(attn_node, proj_out_node)
                c.edge(proj_out_node, add_node)
                
                # Register IO
                layer_io_nodes[idx] = (in_node, add_node)
        
        elif is_multiview_fusion:
            # Create a subgraph for Multiview Fusion
            with dot.subgraph(name=f'cluster_{idx}') as c:
                c.attr(label=f'Layer {idx}: {layer_name}\n{params}\nIn: {shapes["in"]}\nOut: {shapes["out"]}')
                c.attr(style='filled', color='lightyellow')
                
                # Internal nodes
                in_node = f"{idx}_in"
                geo_pe_node = f"{idx}_geo_pe"
                q_proj_node = f"{idx}_q_proj"
                k_proj_node = f"{idx}_k_proj"
                v_proj_node = f"{idx}_v_proj"
                attn_node = f"{idx}_attn"
                out_proj_node = f"{idx}_out_proj"
                add_node = f"{idx}_add"
                
                c.node(in_node, "Input Feats", shape='point')
                c.node(geo_pe_node, "GeoPE\n(Coords)", shape='box', style='filled', fillcolor='lightgreen')
                c.node(q_proj_node, "Q Proj", shape='box', style='filled', fillcolor='white')
                c.node(k_proj_node, "K Proj", shape='box', style='filled', fillcolor='white')
                c.node(v_proj_node, "V Proj", shape='box', style='filled', fillcolor='white')
                c.node(attn_node, "Cross-View\nAttention", shape='diamond', style='filled', fillcolor='orange')
                c.node(out_proj_node, "Out Proj", shape='box', style='filled', fillcolor='white')
                c.node(add_node, "Add\n(Residual)", shape='circle', style='filled', fillcolor='white')
                
                # Internal Edges
                c.edge(in_node, q_proj_node)
                c.edge(in_node, k_proj_node)
                c.edge(in_node, v_proj_node)
                c.edge(in_node, add_node, style='dashed', label='skip')
                
                c.edge(geo_pe_node, q_proj_node, label='add PE', style='dotted')
                c.edge(geo_pe_node, k_proj_node, label='add PE', style='dotted')
                
                c.edge(q_proj_node, attn_node, label='Q')
                c.edge(k_proj_node, attn_node, label='K')
                c.edge(v_proj_node, attn_node, label='V')
                
                c.edge(attn_node, out_proj_node)
                c.edge(out_proj_node, add_node)
                
                # Register IO
                layer_io_nodes[idx] = (in_node, add_node)

        else:
            # Standard Node
            label = f"Layer {idx}: {layer_name}\n"
            if params:
                label += f"{params}\n"
            label += f"In: {shapes['in']}\nOut: {shapes['out']}"
            
            node_id = str(idx)
            dot.node(node_id, label, shape='box', style='filled', fillcolor='white')
            
            # Register IO
            layer_io_nodes[idx] = (node_id, node_id)
        
        # External Edges (Connect from previous layers)
        f = getattr(layer, 'f', -1)
        
        srcs = []
        if f == -1:
            srcs = [idx - 1]
        elif isinstance(f, list):
            srcs = f
        else: # int
            srcs = [f]
            
        for src in srcs:
            # Handle -1 inside list
            if src == -1 and isinstance(f, list):
                real_src = idx - 1
            else:
                real_src = src
            
            # Map -1 to -1 (input) if solitary
            if real_src == -1 and idx == 0:
                 # Special case: first layer input
                 pass # handled below by lookup
            elif real_src == -1:
                real_src = idx - 1

            # Get Source Output Node
            # If real_src is -1, it maps to input
            # If real_src >= 0, it maps to that layer
            
            src_out_id = layer_io_nodes.get(real_src, (None, None))[1]
            dst_in_id = layer_io_nodes[idx][0]
            
            if src_out_id and dst_in_id:
                 # If connecting to/from cluster, graphviz usually handles node-to-node fine.
                 # lhead/ltail are needed if we want edge to stop at cluster border, 
                 # but connecting to internal node is often clearer for detailed view.
                 dot.edge(src_out_id, dst_in_id)

    try:
        # Save source first
        dot.save(output_file + '.gv')
        print(f"Saved graph source to {output_file}.gv")
        
        output_path = dot.render(output_file, view=False)
        print(f"Successfully rendered architecture to {output_path}")
    except Exception as e:
        print(f"Error rendering graph (likely missing 'dot' executable): {e}")
        print(f"Graph source saved to {output_file}.gv. You can render it using 'dot -Tpng {output_file}.gv -o {output_file}.png' if you install graphviz.")

if __name__ == "__main__":
    main()
