from graphviz import Digraph

# Initialize diagram
dfd = Digraph(format="png")
dfd.attr(rankdir="TB")

# External entities
dfd.node("Users", shape="box", style="filled",
         color="lightpink", label="Users")
dfd.node("Social Media Advertising System", shape="ellipse", style="filled",
         color="lightblue", label="Social Media Advertising System")
dfd.node("Offer Conversion System", shape="ellipse", style="filled",
         color="lightgreen", label="Offer Conversion System")
dfd.node("Conversion", shape="box", style="filled",
         color="lightyellow", label="Conversion")

# Data flows
dfd.edge("Users", "Social Media Advertising System",
         label="See Ad", color="blue")
dfd.edge("Social Media Advertising System", "Offer Conversion System",
         label="Transition to Offer", color="blue")
dfd.edge("Offer Conversion System", "Conversion",
         label="Complete Offer", color="orange")
dfd.edge("Conversion", "Users", xlabel="Provide Offer", color="green")

# Displaying analytics (metrics)
dfd.node("Analytics", shape="box", style="dashed",
         color="gray", label="Analytics & Metrics")
dfd.edge("Social Media Advertising System", "Analytics",
         label="Display Social Media Ad Metrics", color="purple")
dfd.edge("Offer Conversion System", "Analytics",
         label="Display Offer Conversion Metrics", color="purple")

# Generate the diagram
diagram_path = "context_diagram_social_media_offer_conversion"
dfd.render(diagram_path)

# Return the path to the image
diagram_path + ".png"
