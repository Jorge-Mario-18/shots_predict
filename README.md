# Goal Prediction Machine

Welcome to the **Goal Prediction Machine**! 

This tool uses a **GradientBoostingClassifier** model, powered by data from [StatsBomb](https://statsbomb.com), to predict the likelihood of a goal based on a specific location on the soccer field. 

## Field Dimensions
- The field dimensions are **120 x 80**.

## How to Use
1. Input the **coordinates** on the field to select a spot accurately.
2. Provide the following additional inputs for more precise predictions:
   - **Under Pressure**: Indicate whether the shot was taken under pressure.
   - **Type of Shot**: Specify the type of shot taken.
   - **Preceding Action**: Select the action that led to the shot.

Use these inputs to refine the predictions and gain deeper insights into goal-scoring chances!

---
**Built with**: Python, [Streamlit](https://streamlit.io/), and [mplsoccer](https://mplsoccer.readthedocs.io/)
