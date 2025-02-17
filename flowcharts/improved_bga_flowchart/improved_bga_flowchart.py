from graphviz import Digraph

# Global font settings (same as your SA flowchart)
FONT_NAME = "Helvetica-Bold"
FONT_SIZE = "32"

# Create a new directed graph with A4 page settings (portrait: 8.27 x 11.69 inches)
dot = Digraph(comment='Improved BGA Flowchart', format='png')

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

# Helper dictionaries for node shapes
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
# Define Nodes with consistent pastel styling and technical details
###############################################################################

# Start / End
dot.node('start', 'Start', **oval_box, fillcolor='#AEDFF7', fontcolor='black')
dot.node('end', 'End\nReturn best feasible solution', **oval_box, fillcolor='#AEDFF7', fontcolor='black')

# Initialization
dot.node('pseudo_init', 
         'Pseudo-random Initialization\n(Chu & Beasley [1])',
         **default_box, fillcolor='#85C1E9', fontcolor='black')

# Generation Loop
dot.node('gen_loop', 
         'Generation Loop\n(1 to max_generations)',
         **default_box, fillcolor='#A9DFBF', fontcolor='black')

# Stochastic Ranking Sort (first pass)
dot.node('stoch_sort', 
         'Stochastic Ranking Sort\n(Runarsson & Yao [2])',
         **default_box, fillcolor='#F7DC6F', fontcolor='black')

# Offspring Production Subprocess
dot.node('offspring_loop',
         'Produce Offspring\n(pop_size // 2 pairs)',
         **default_box, fillcolor='#F0B27A', fontcolor='black')
dot.node('uniform_xover', 
         'Uniform Crossover\n(prob = crossover_rate)',
         **default_box, fillcolor='#BB8FCE', fontcolor='black')
dot.node('adaptive_mutation', 
         'Adaptive Mutation\n(base_mutation_rate,\nadaptive_mutation_threshold)',
         **default_box, fillcolor='#BB8FCE', fontcolor='black')
dot.node('heuristic_improvement', 
         'Heuristic Improvement\n(DROP/ADD)',
         **default_box, fillcolor='#A3E4D7', fontcolor='black')

# Decision: Is offspring population full?
dot.node('offspring_decision', 
         'Offspring < pop_size?',
         **diamond_box, fillcolor='#F7DC6F', fontcolor='black')

# Combine and re-rank
dot.node('combine_pop', 
         'Combine Parents + Offspring\n(2×pop_size)',
         **default_box, fillcolor='#F5B7B1', fontcolor='black')
dot.node('stoch_sort2', 
         'Stochastic Ranking Again\n(keep top pop_size)',
         **default_box, fillcolor='#76D7C4', fontcolor='black')

# Update Best Feasible Solution
dot.node('update_best',
         'Update Best Feasible Solution\n(if feasible & cost improved)',
         **default_box, fillcolor='#F9E79F', fontcolor='black')

# Decision: Continue Generations?
dot.node('gen_decision',
         'More Generations?',
         **diamond_box, fillcolor='#F9E79F', fontcolor='black')

###############################################################################
# Define Edges (Flow of the Algorithm with Decision Criteria)
###############################################################################

# Main flow: Start → Initialization → Generation Loop
dot.edge('start', 'pseudo_init')
dot.edge('pseudo_init', 'gen_loop')

# Generation loop: Ranking step then offspring production
dot.edge('gen_loop', 'stoch_sort', label='Rank Population')
dot.edge('stoch_sort', 'offspring_loop', label='Proceed to Offspring')

# Offspring production subprocess
dot.edge('offspring_loop', 'uniform_xover', label='Select Parents')
dot.edge('uniform_xover', 'adaptive_mutation', label='Crossover')
dot.edge('adaptive_mutation', 'heuristic_improvement', label='Mutate')
dot.edge('heuristic_improvement', 'offspring_decision', label='Improve')

# Offspring decision: Are we done producing offspring?
dot.edge('offspring_decision', 'offspring_loop', label='Not Full', style='dashed')
dot.edge('offspring_decision', 'combine_pop', label='Full', style='bold')

# Combine and re-rank the population
dot.edge('combine_pop', 'stoch_sort2', label='Combine')
dot.edge('stoch_sort2', 'update_best', label='Rank Combined')

# Update best solution, then decide on next generation
dot.edge('update_best', 'gen_decision', label='Evaluate')
dot.edge('gen_decision', 'gen_loop', label='Yes: Continue', style='dashed')
dot.edge('gen_decision', 'end', label='No: Stop', style='bold')

###############################################################################
# Subgraph for Offspring Production (dashed blue outline, no label)
###############################################################################
with dot.subgraph(name='cluster_offspring') as c:
    c.attr(style='dashed', color='blue')
    for n in ['offspring_loop', 'uniform_xover', 'adaptive_mutation', 'heuristic_improvement', 'offspring_decision']:
        c.node(n)

# Render the flowchart to a PNG file and open it.
dot.render('improved_bga_flowchart', view=True)