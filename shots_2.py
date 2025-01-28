import streamlit as st
import matplotlib as plt
import pandas as pd
import numpy as np
import pickle 
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from mplsoccer import Pitch
from matplotlib.patches import FancyArrow


st.title("_Predict Soccer_ :blue[Goals] :soccer:")

st.markdown("""Welcome to the Goal Prediction Machine! This tool uses a 
            GradientBoostingClassifier model, powered by data from StatsBomb, 
            to predict the likelihood of a goal based on a specific location on the 
            field. The field dimensions are 120 x 80. Simply input the coordinates 
            on the field to select a spot accurately. Additionally, there are other 
            inputs to consider: """)

st.markdown(" -Under Pressure: Whether the shot was taken under pressure.")
st.markdown(" -Type of Shot: The type of shot taken.")
st.markdown(" -Preceding Action: The action that led to the shot.")
st.markdown(" Use these inputs to get more precise predictions.")


# Handle X-coordinate input safely
shot_loaction_x_input = st.text_input("X - Coordinate", "119")
try:
    shot_loaction_x = float(shot_loaction_x_input) if shot_loaction_x_input.strip() else 119.0
except ValueError:
    st.error("Invalid input for X-Coordinate! Using default value of 119.")
    shot_loaction_x = 119.0

# Handle Y-coordinate input safely
shot_loaction_y_input = st.text_input("Y - Coordinate", "40")
try:
    shot_loaction_y = float(shot_loaction_y_input) if shot_loaction_y_input.strip() else 40.0
except ValueError:
    st.error("Invalid input for Y-Coordinate! Using default value of 40.")
    shot_loaction_y = 40.0


def plot_pitch_with_point_and_arrow():
    """
    Creates a soccer pitch with a point and an arrow pointing towards (120, 40).
    
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """

    # Soccer pitch setup
    pitch = Pitch(
        positional=True,
        pitch_color='grass',
        line_color='white',
        label=True,
        tick=True,
        stripe=True  # Optional stripes
    )
    fig, ax = pitch.draw()

    # Starting point (where the arrow originates)
    x_start, y_start = shot_loaction_x, shot_loaction_y  # Example starting point
    target_x, target_y = 120, 40  # Center of the goal

    # Calculate direction vector (dx, dy) for the arrow
    dx = target_x - x_start
    dy = target_y - y_start

    # Normalize the vector to maintain the arrow length
    arrow_length = 20  # Adjust arrow length as needed
    norm = np.sqrt(dx**2 + dy**2)
    dx_normalized = (dx / norm) * arrow_length
    dy_normalized = (dy / norm) * arrow_length



    # Plot the arrow using the normalized direction
    arrow = FancyArrow(
        x_start, y_start, dx_normalized, dy_normalized,
        width=1, color='blue', edgecolor='black',
        head_width=2.5, head_length=2.5, length_includes_head=True
    )
    ax.add_patch(arrow)
    # Plot the point
    ax.scatter(x_start, y_start, color='red', s=100, label='Point', edgecolor="black")

    plt.tight_layout()
    return fig

# Generate and display the plot
fig = plot_pitch_with_point_and_arrow()
st.pyplot(fig)



model = pickle.load(open('clf_2.pkl', 'rb'))
under_pressure = st.selectbox("Under Pressure",(True, False))
shot_type_mapping = {"Open Play": 1, "Free Kick": 2, "Penalty": 3, "Corner": 3}
shot_type = st.selectbox("Shot Type", shot_type_mapping.keys())
shot_type_code = shot_type_mapping[shot_type]
action_mapping = {"Pass": 1, "Carry": 2, "Shot": 3}
preceding_action = st.selectbox("Preceding Action", action_mapping.keys())
preceding_action_code = action_mapping[preceding_action]
st.text("")
st.text("")
st.text("")
col1, col2, col3, col4, col5 = st.columns(5)
if col3.button("Predict  :soccer:"):
    x_start, y_start = shot_loaction_x, shot_loaction_y
    features = [[preceding_action_code, shot_type_code, under_pressure, shot_loaction_x, shot_loaction_y]]
    prediction = model.predict(features)
    output = prediction
    st.divider()
    col5, col7, col8 = st.columns(3)
    prob = model.predict_proba(features)
    if output == 1:
        text = "GOAL!"
        img = "images.jpg"
    else:
        text = "It's a miss!"
        img = "messi.jpg"
    col7.title(text)
    col7.image(img)
    
    col7.text(f"From position ({x_start}, {y_start})")
    proba_output = model.predict_proba(features)
    class_0_proba = proba_output[0][0]  # Probability for class 0
    class_1_proba = proba_output[0][1]
    st.write("**Prediction Probabilities:**")
    st.write(f"Class 0 (No Goal): {class_0_proba:.2%}")
    st.progress(class_0_proba)  # Visual progress bar for Class 0

    st.write(f"Class 1 (Goal): {class_1_proba:.2%}")
    st.progress(class_1_proba)  # Visual progress bar for Class 1