import dash
from dash import html, dcc
from dash import no_update
from dash.dependencies import Input, Output, State
from predict import predict_heart_disease 
import plotly.express as px
import pandas as pd
from functools import lru_cache

app = dash.Dash(__name__)
app.title = "SafeHeart"
server = app.server

@lru_cache(maxsize=1)
def load_data():
    return pd.read_csv('CVD_cleaned_v2.csv')

# Dropdown options
sex_options = ['Male', 'Female']
age_options = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
               '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
yes_no_options = ['Yes', 'No']
age_order = [
    '18-24', '25-29', '30-34', '35-39', '40-44',
    '45-49', '50-54', '55-59', '60-64', '65-69',
    '70-74', '75-79', '80+'
]

app.layout = html.Div([
    html.H1("SafeHeart Dashboard", style={"textAlign": "center"}),

    dcc.Tabs([
        dcc.Tab(label='🧠 Predict Heart Disease Risk', children=[
            html.Br(),
            html.Div([
                html.Div([
                    html.Label("Sex:"),
                    dcc.Dropdown(["Male", "Female"], id="sex", placeholder="Select...")
                ]),

                html.Div([
                    html.Label("Age Category:"),
                    dcc.Dropdown(sorted(load_data()['Age_Category'].unique()), id="age", placeholder="Select...")
                ]),

                html.Div([
                    html.Label("Weight (kg):"),
                    dcc.Input(type="number", id="weight", placeholder="Enter weight")
                ]),

                html.Div([
                    html.Label("Height (cm):"),
                    dcc.Input(type="number", id="height", placeholder="Enter height")
                ]),

                html.Div([
                    html.Label("BMI:"),
                    dcc.Input(type="number", id="bmi", placeholder="Enter BMI")
                ]),

                html.Div([
                    html.Label("Diabetes:"),
                    dcc.Dropdown(["Yes", "No"], id="diabetes", placeholder="Select...")
                ]),

                html.Div([
                    html.Label("Arthritis:"),
                    dcc.Dropdown(["Yes", "No"], id="arthritis", placeholder="Select...")
                ]),

                html.Br(),
                html.Button('Predict Risk', id='submit-button', n_clicks=0),
                html.Div(id='prediction-output'),
                dcc.Loading(
                    type="circle",
                    children=html.Div([
                        html.Div(id="overall-plot-container", children=[
                            html.H4("📊 Overall BMI vs Age Category"),
                            dcc.Graph(id="overall-plot")
                        ], style={"display": "none"}),

                        html.Div(id="sex-specific-plot-container", children=[
                            html.H4("📊 Sex-specific BMI vs Age Category"),
                            dcc.Graph(id="sex-specific-plot")
                        ], style={"display": "none"})
                    ])
                )
            ], style={"width": "50%", "margin": "auto"})
        ]),

        dcc.Tab(label='📊 Data Visualizations', children=[
            html.Br(),
            html.Div([
                html.H3("Heart Disease Counts by Sex"),
                dcc.Graph(figure=px.histogram(load_data(), x='Sex', color='Heart_Disease', barmode='group')),

                html.H3("BMI Distribution"),
                dcc.Graph(figure=px.histogram(load_data(), x='BMI', nbins=40)),

                html.H3("Age Category Distribution"),
                dcc.Graph(figure=px.histogram(load_data(), x='Age_Category', category_orders={"Age_Category": age_order})),

                html.H3("BMI vs Age Category colored by Diabetes"),
                dcc.Graph(figure=px.box(load_data(), x='Age_Category', y='BMI', color='Diabetes', category_orders={"Age_Category": age_order})),

                html.H3("BMI vs Age Category colored by Arthritis"),
                dcc.Graph(figure=px.box(load_data(), x='Age_Category', y='BMI', color='Arthritis', category_orders={"Age_Category": age_order})),
            ], style={"width": "90%", "margin": "auto"})
        ])
    ])
])

# Prediction Callback
@app.callback(
    [
        Output("prediction-output", "children"),
        Output("overall-plot", "figure"),
        Output("sex-specific-plot", "figure"),
        Output("overall-plot-container", "style"),
        Output("sex-specific-plot-container", "style")
    ],
    Input('submit-button', 'n_clicks'),
    State('sex', 'value'),
    State('age', 'value'),
    State('height', 'value'),
    State('weight', 'value'),
    State('bmi', 'value'),
    State('diabetes', 'value'),
    State('arthritis', 'value')
)
def update_prediction_and_plots(n_clicks, sex, age, weight, height, bmi, diabetes, arthritis):
    if n_clicks == 0:
        return "", no_update, no_update, {"display": "none"}, {"display": "none"}

    user_input = {
        "Sex": sex,
        "Age_Category": age,
        "Weight": weight,
        "Height": height,
        "BMI": bmi,
        "Diabetes": diabetes,
        "Arthritis": arthritis
    }

    result = predict_heart_disease(user_input)

    prediction = result['prediction']
    probability = result['probability']

    # Risk label
    risk_text = "🫀 High Risk of Heart Disease" if prediction == 1 else "💚 Low Risk of Heart Disease"
    prediction_output = html.Div([
        html.H3(risk_text),
        html.P(f"Predicted Probability: {probability * 100:.2f}%")
    ])

    # Create visualizations
    display_df = load_data().copy()
    display_df["Heart_Disease_Label"] = display_df["Heart_Disease"].map({0: "No Heart Disease", 1: "Heart Disease"})

    color_map = {
        "Heart Disease": "red",
        "No Heart Disease": "blue"
    }
    overall_fig = px.scatter(
        display_df,
        x="BMI",
        y="Age_Category",
        height=800,
        width=1000,
        color="Heart_Disease_Label",
        facet_col="Heart_Disease_Label",  # This splits charts
        opacity=0.5
    )
    overall_fig.add_scatter(
        x=[bmi], y=[age],
        mode="markers",
        marker=dict(color="gold", size=25, symbol="star"),
        name="You",
        showlegend=True
    )

    # Sex-specific plot
    filtered_df = display_df[display_df["Sex"] == sex]
    sex_fig = px.scatter(
        filtered_df,
        x="BMI",
        y="Age_Category",
        height=800,
        width= 1000,
        color="Heart_Disease_Label",
        category_orders={"Age_Category": age_order}, 
        color_discrete_map={"Heart Disease": "red", "No Heart Disease": "blue"},
        title=f"BMI vs Age (Sex: {sex})",
        opacity=0.2,
        size_max=8,
        render_mode="webgl"
    )
    sex_fig.add_scatter(
        x=[bmi], y=[age],
        mode="markers",
        marker=dict(color="gold", size=25, symbol="star"),
        name="You",
        showlegend=True
    )

    return prediction_output, overall_fig, sex_fig, {"display": "block"}, {"display": "block"}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)