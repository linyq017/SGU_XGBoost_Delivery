import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set plot styles
sns.set_style("ticks")
sns.set_context("paper")

def load_xgb_model(model_path):
    """Load the pre-trained XGBoost model from a file."""
    print("Loading XGBoost model...")
    return xgb.Booster(model_file=model_path)

def load_test_data(file_path):
    """Load test dataset and preprocess feature and target columns."""
    print("Loading test dataset...")
    test = pd.read_csv(file_path, sep=",", decimal=".")
    X_test = test.drop(columns=["REGION", "GENERAL_TX", "KARTTYP", "QD_GENERAL_TX"], axis=1)
    y_test = LabelEncoder().fit_transform(test["GENERAL_TX"])
    print(f"Test data loaded with {X_test.shape[0]} samples and {X_test.shape[1]} features.")
    return X_test, y_test

def compute_shap_values(model, X_test):
    """Compute SHAP values for the given test dataset."""
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("SHAP values computed.")
    return shap_values

def generate_summary_plot(shap_values, X_test, feature_names, classes, colors, output_path, max_display_features=15):
    """Generate and save a SHAP summary plot with a custom colormap."""
    print("Generating SHAP summary plot...")

    # Determine class ordering based on SHAP value magnitude
    class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
    cmap = plt_colors.ListedColormap(np.array(colors)[class_inds])

    # Generate SHAP summary plot
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=feature_names, 
        color=cmap, 
        class_names=classes, 
        show=False,  # Prevent immediate display
        max_display=max_display_features, 
        plot_size=(8.2, 5)
    )

    # Adjust tick label font sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Ensure tight layout
    plt.tight_layout()

    # Set x-axis label
    plt.xlabel("Mean SHAP value (average impact on XGBoost output magnitude)", fontsize=9.5)

    # Save the plot
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"SHAP summary plot saved to {output_path}")

    # Show plot
    plt.show()

if __name__ == "__main__":
    # Define file paths
    model_path = "/workspace/data/SGU/SFSI/SFSI/XBG10x_akermark_7class/20240417095539all_xy/best_model.json"
    test_data_path = "/workspace/data/Akermark/MASTER_test_plotxy.csv"
    output_plot_path = "/workspace/data/manuscript1plots/shap_summary_plot.png" 
    # Load model and data
    xgb_model = load_xgb_model(model_path)
    X_test, y_test = load_test_data(test_data_path)

    # Compute SHAP values
    shap_values = compute_shap_values(xgb_model, X_test)

    # Define class names and colors
    classes = ["Coarse sed FOR", "Coarse sed AGR", "Fine sed FOR", "Fine sed AGR", "Peat", "Rock outcrops", "Till"]
    colors = ["#6F8C57", "#446589", "#EFD460", "#B0633F", "#926CCF", "#EA5A94", "#9C9C9C"]

    # Define feature names
    feature_display_names = [
        "Easting", "Northing", "Digital Elevation Model", "Elevation above Stream (1ha network)", 
        "Elevation above Stream (10ha network)", "Downslope Index 2m drop", "Circular Variance of Aspect", 
        "Standard Deviation of Slope", "Deviation from Mean Elevation", "Terrain Ruggedness Index", 
        "National Land Cover Map", "SGU QD Maps base layer", "Distance to the Highest Coastline", 
        "Depth to Bedrock", "Age from Deglaciation", "Multiscale Roughness Magnitude", 
        "Maximum Elevation Deviation", "Circular Variance of Aspect 20m", "Circular Variance of Aspect 50m", 
        "Maximum Curvature 20m", "Minimum Curvature 20m", "Slope 20m", "Slope 50m", 
        "Maximum Elevation Deviation 20m", "Maximum Elevation Deviation 50m", 
        "Average Normal Vector Angular Deviation 20m", "Directional Relief", "Downslope Index with 2m drop 20m", 
        "Geomorphons 20m", "Max Downslope Elevation Change 20m", "Normalized Difference Vegetation Index", 
        "Profile Curvature 20m", "Relative Topographic Positions 20m", "Topographic Wetness Index 20m"
    ]

    # Generate and save SHAP summary plot
    generate_summary_plot(shap_values, X_test, feature_display_names, classes, colors, output_plot_path)
