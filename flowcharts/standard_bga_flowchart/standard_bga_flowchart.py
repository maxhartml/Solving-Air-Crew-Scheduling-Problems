from graphviz import Digraph

# Global font settings
FONT_NAME = "Helvetica-Bold"
FONT_SIZE = "30"

# Create a new directed graph with A4 page settings (portrait: 8.27 x 11.69 inches)
dot = Digraph(comment='Standard BGA Flowchart', format='png')

dot.attr(
    rankdir='TB',
    splines='spline',
    nodesep='0.8',
    ranksep='1.5',
    bgcolor='white',
    size="8.27,11.69!",
    page="8.27,11.69",
    margin="0.3"
)

# Set global attributes for nodes and edges
dot.attr('node', fontname=FONT_NAME, fontsize=FONT_SIZE)
dot.attr('edge', fontname=FONT_NAME, fontsize=FONT_SIZE, fontcolor='black', penwidth='3')

# Helper dictionaries for node shapes (reduced width for a more vertical layout)
default_box = {
    'shape': 'box',
    'style': 'rounded,filled',
    'width': '5',
    'height': '2'
}
diamond_box = {
    'shape': 'diamond',
    'style': 'filled',
    'width': '5',
    'height': '2'
}
oval_box = {
    'shape': 'oval',
    'style': 'filled',
    'width': '3',
    'height': '2'
}

###############################################################################
# Define Nodes with Technical Details and Decision Points
###############################################################################

# Start / End
dot.node('start', 'Start', **oval_box, fillcolor='#AEDFF7', fontcolor='black')
dot.node('end', 'End\nReturn best solution & fitness', **oval_box, fillcolor='#AEDFF7', fontcolor='black')

# Initialization
dot.node('init', 'Initialize Population\n(pop_size, num_cols)', **default_box, fillcolor='#85C1E9', fontcolor='black')

# Generation Loop
dot.node('gen_loop', 'Generation Loop\n(max_generations)', **default_box, fillcolor='#A9DFBF', fontcolor='black')

# Offspring Generation Sub-process
dot.node('offspring_loop', 'Offspring Generation\n(while new_population < pop_size)', **default_box, fillcolor='#F7DC6F', fontcolor='black')
dot.node('tournament', 'Tournament Selection\n(k = tournament_k)', **default_box, fillcolor='#F0B27A', fontcolor='black')
dot.node('crossover', 'One-Point Crossover\n(prob = crossover_rate)', **default_box, fillcolor='#BB8FCE', fontcolor='black')
dot.node('mutation', 'Mutation\n(prob = mutation_rate)', **default_box, fillcolor='#BB8FCE', fontcolor='black')
dot.node('add_offspring', 'Add Offspring\nto new_population', **default_box, fillcolor='#A3E4D7', fontcolor='black')

# Decision: Is the offspring population full?
dot.node('offspring_decision', 'New population full?', **diamond_box, fillcolor='#F7DC6F', fontcolor='black')

# Post-Offspring: Combine and Re-rank
dot.node('trim', 'Trim new_population\nto pop_size', **default_box, fillcolor='#F5B7B1', fontcolor='black')
dot.node('recompute', 'Recompute Fitness\n(penalty_factor)', **default_box, fillcolor='#76D7C4', fontcolor='black')
dot.node('update', 'Update Best Solution\n(if improved)', **default_box, fillcolor='#76D7C4', fontcolor='black')

# Decision: More Generations?
dot.node('gen_decision', 'More generations?', **diamond_box, fillcolor='#F9E79F', fontcolor='black')

###############################################################################
# Define Edges with Detailed Labels for Clarity
###############################################################################

# Main Flow: Start -> Initialization -> Generation Loop
dot.edge('start', 'init')
dot.edge('init', 'gen_loop', label='Initial Population')

# Generation Loop: Enter Offspring Generation
dot.edge('gen_loop', 'offspring_loop', label='Begin Offspring Production')

# Offspring Production Process
dot.edge('offspring_loop', 'tournament', label='Select Parents')
dot.edge('tournament', 'crossover', label='Crossover')
dot.edge('crossover', 'mutation', label='Apply Mutation')
dot.edge('mutation', 'add_offspring', label='Create Offspring')
dot.edge('add_offspring', 'offspring_decision', label='Check Offspring Count')

# Decision: Offspring Population Full?
dot.edge('offspring_decision', 'offspring_loop', label='No', style='dashed')  # Not full, continue producing
dot.edge('offspring_decision', 'trim', label='Yes', style='bold')           # Full, proceed

# Post-Offspring: Combine and Re-rank
dot.edge('trim', 'recompute', label='Trim Population')
dot.edge('recompute', 'update', label='Recalculate Fitness')
dot.edge('update', 'gen_decision', label='Evaluate Best')

# Decision: Continue Generations?
dot.edge('gen_decision', 'gen_loop', label='Yes: Continue', style='dashed')
dot.edge('gen_decision', 'end', label='No: Finish', style='bold')

###############################################################################
# Subgraph: Group Offspring Production Process (Dashed Blue Outline)
###############################################################################
with dot.subgraph(name='cluster_offspring') as c:
    c.attr(style='dashed', color='blue')
    for n in ['offspring_loop', 'tournament', 'crossover', 'mutation', 'add_offspring', 'offspring_decision']:
        c.node(n)

# Render the flowchart to a PNG file and open it.
dot.render('standard_bga_flowchart', view=True)