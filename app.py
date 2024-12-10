import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# Define the CPDs (Conditional Probability Distributions)
cpd_guest = TabularCPD('Guest', 3, [[0.33], [0.33], [0.33]])
cpd_price = TabularCPD('Price', 3, [[0.33], [0.33], [0.33]])

cpd_host = TabularCPD('Host', 3, 
                     [[0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
                      [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
                      [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]], 
                     evidence=['Guest', 'Price'], evidence_card=[3, 3])

# Define the Bayesian Network structure
model = BayesianNetwork([('Guest', 'Host'), ('Price', 'Host')])

# Add the CPDs to the model
model.add_cpds(cpd_guest, cpd_price, cpd_host)

# Check if the model is valid
if model.check_model():
    st.write("Model is valid!")
else:
    st.write("Model is invalid!")

# Inference using Variable Elimination
infer = VariableElimination(model)

# Perform a query with evidence
posterior_p = infer.query(['Host'], evidence={'Guest': 2, 'Price': 2})
st.write(posterior_p)

# Manually create a NetworkX graph from the Bayesian Network
model_graph = nx.DiGraph()  # Directed graph for Bayesian Network

# Add nodes from the model
model_graph.add_nodes_from(model.nodes())

# Add edges from the model
model_graph.add_edges_from(model.edges())

# Create a figure and axis for plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Visualize the model using NetworkX
nx.draw(model_graph, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold', arrows=True, ax=ax)
ax.set_title('Bayesian Network Structure')

# Display the plot in Streamlit
st.pyplot(fig)

# Save the plot to a file
plt.savefig('Final-output.png')
