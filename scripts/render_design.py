import os

def render_design():
    dot_content = """
digraph YOLO12_Multiview {
    rankdir=TD;
    compound=true;
    node [shape=box, style=filled, fillcolor=white];

    subgraph cluster_inputs {
        label="Inputs";
        style=filled;
        color=lightgrey;
        Img1 [label="View 1 Image"];
        Img2 [label="View 2 Image"];
        Coord1 [label="View 1 Coords"];
        Coord2 [label="View 2 Coords"];
    }

    subgraph cluster_backbone {
        label="Shared Backbone";
        style=filled;
        color=lightblue;
        Backbone [label="Backbone"];
        Feat1 [label="Feature Map 1"];
        Feat2 [label="Feature Map 2"];
    }

    subgraph cluster_fusion {
        label="Fusion Layer (P4/P5)";
        style=filled;
        color=lightyellow;
        GeoPE [label="GeoPE"];
        PosEmb1 [label="PosEmb1"];
        PosEmb2 [label="PosEmb2"];
        Q1 [label="Q1: Feat+Pos"];
        K_All [label="K: All Feats+Pos"];
        V_All [label="V: All Feats"];
        Attention [label="Attention", shape=diamond, fillcolor=orange];
        FusedFeat1 [label="FusedFeat1"];
    }

    Head [label="Detection Head"];

    # Edges
    Img1 -> Backbone;
    Img2 -> Backbone;
    Coord1 -> GeoPE;
    Coord2 -> GeoPE;

    Backbone -> Feat1;
    Backbone -> Feat2;

    GeoPE -> PosEmb1;
    GeoPE -> PosEmb2;

    Feat1 -> Q1;
    PosEmb1 -> Q1;
    
    Feat2 -> K_All;
    PosEmb2 -> K_All;
    
    Feat2 -> V_All;

    Q1 -> Attention;
    K_All -> Attention;
    V_All -> Attention;
    
    Attention -> FusedFeat1;

    FusedFeat1 -> Head;
}
"""
    output_dot = 'yolo12_multiview_design.dot'
    output_png = 'yolo12_multiview_design.png'
    
    with open(output_dot, 'w') as f:
        f.write(dot_content)
        
    print(f"DOT file saved to {os.path.abspath(output_dot)}")
    
    cmd = f"dot -Tpng {output_dot} -o {output_png}"
    print(f"Running: {cmd}")
    os.system(cmd)
    
    if os.path.exists(output_png):
        print(f"Graph rendered to {os.path.abspath(output_png)}")
    else:
        print("Error: PNG file was not created.")

if __name__ == '__main__':
    render_design()
