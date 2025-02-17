from graphviz import Digraph

# Global font settings
FONT_NAME = "Helvetica-Bold"
FONT_SIZE = "32"

# Create a new directed graph with A4 page settings (portrait: 8.27 x 11.69 inches)
dot = Digraph(comment='Simulated Annealing Flowchart', format='png')

dot.attr(
    rankdir='TB',
    splines='spline',
    nodesep='1.2',
    ranksep='1.2',
    bgcolor='white',
    size="8.27,11.69!",
    page="8.27,11.69",
    margin="0.3"
)

# Set global attributes for nodes and edges
dot.attr('node', fontname=FONT_NAME, fontsize=FONT_SIZE)
dot.attr('edge', fontname=FONT_NAME, fontsize=FONT_SIZE, fontcolor='black', penwidth='3')

# Helper dictionaries for node shapes (to reduce redundancies)
default_box = {
    'shape': 'box',
    'style': 'rounded,filled',
    'width': '6',
    'height': '2'
}
diamond_box = {
    'shape': 'diamond',
    'style': 'filled',
    'width': '6',
    'height': '3'
}
oval_box = {
    'shape': 'oval',
    'style': 'filled',
    'width': '3',
    'height': '2'
}

###############################################################################
# Define Nodes with Consistent Styling and Detailed Descriptions
###############################################################################

dot.node('start', 'Start', **oval_box, fillcolor='#AEDFF7', fontcolor='black')

dot.node('init', 
         'Initialize solution x randomly\nCompute F(x) = cost + penalty_factor * (coverage violations)',
         **default_box, fillcolor='#85C1E9', fontcolor='black')

dot.node('best', 
         'Set initial best solution:\n  x_best ← x\n  F_best ← F(x)',
         **default_box, fillcolor='#85C1E9', fontcolor='black')

dot.node('iter', 
         'For each iteration\n(1 to max_iter)',
         **default_box, fillcolor='#A9DFBF', fontcolor='black')

dot.node('neighbor', 
         "Generate neighbor x'\n(by flipping one random bit)",
         **default_box, fillcolor='#BB8FCE', fontcolor='black')

dot.node('compute', 
         "Compute F(x')",
         **default_box, fillcolor='#BB8FCE', fontcolor='black')

dot.node('compare', 
         "Is F(x') < F(x)?",
         **diamond_box, fillcolor='#F7DC6F', fontcolor='black')

dot.node('accept', 
         "Accept x' as current solution",
         **default_box, fillcolor='#A3E4D7', fontcolor='black')

dot.node('metropolis', 
         "Δ = F(x') - F(x)\nAcceptance probability = exp(-Δ/T)\nAccept x' with this probability?",
         **default_box, fillcolor='#F0B27A', fontcolor='black')

dot.node('update_best', 
         "If F(current) < F_best\nUpdate best solution",
         **default_box, fillcolor='#F5B7B1', fontcolor='black')

dot.node('cool', 
         "Update temperature:\nT = α × T",
         **default_box, fillcolor='#76D7C4', fontcolor='black')

dot.node('loop', 
         "More iterations?\n(Iteration < max_iter & T > T_min)",
         **diamond_box, fillcolor='#F9E79F', fontcolor='black')

dot.node('end', 
         "End\nReturn x_best and F_best",
         **oval_box, fillcolor='#AEDFF7', fontcolor='black')

###############################################################################
# Define Edges with Detailed Labels to Clarify the Flow and Criteria
###############################################################################

dot.edge('start', 'init')
dot.edge('init', 'best', label="Initialize & Compute F(x)")
dot.edge('best', 'iter', label="Set initial best")
dot.edge('iter', 'neighbor')
dot.edge('neighbor', 'compute', label="Generate neighbor solution")
dot.edge('compute', 'compare', label="Evaluate F(x')")
dot.edge('compare', 'accept', label='Yes:\nF(x\') < F(x)')
dot.edge('compare', 'metropolis', label='No:\nF(x\') ≥ F(x)')
dot.edge('accept', 'update_best')
dot.edge('metropolis', 'update_best')
dot.edge('update_best', 'cool', label="If improved, update best")
dot.edge('cool', 'loop')
dot.edge('loop', 'iter', label='Yes:\nContinue iterations', style='dashed')
dot.edge('loop', 'end', label='No:\nTerminate SA', style='bold')

###############################################################################
# Subgraph: Group the Iterative Process (Dashed Blue Outline, No Label)
###############################################################################
with dot.subgraph(name='cluster_iteration') as c:
    c.attr(style='dashed', color='blue')
    for n in ['iter', 'neighbor', 'compute', 'compare', 'accept', 'metropolis', 'update_best', 'cool', 'loop']:
        c.node(n)

# Render the flowchart to a PNG file and open it.
dot.render('simulated_annealing_flowchart', view=True)