# Recipe Recommendation System

A machine learning system that recommends similar recipes based on nutritional content using K-Nearest Neighbors algorithm.

## Overview

This project implements a recipe recommendation engine that finds similar recipes based on nutritional profiles. By analyzing features like calories, fat content, carbohydrates, protein, and other nutritional values, the system can suggest recipes that have similar nutritional characteristics to a reference recipe.

The recommendation engine uses the K-Nearest Neighbors algorithm with cosine similarity to find the most nutritionally similar recipes in the dataset.

## Dataset

The project uses a comprehensive recipes dataset with the following key columns:
- `RecipeId`: Unique identifier for each recipe
- `Name`: Recipe name
- `CookTime`: Time required to cook the recipe
- `PrepTime`: Time required for preparation
- `TotalTime`: Total time required for the recipe
- `RecipeIngredientParts`: List of ingredients
- Nutritional information columns:
  - `Calories`
  - `FatContent`
  - `SaturatedFatContent`
  - `CholesterolContent`
  - `SodiumContent`
  - `CarbohydrateContent`
  - `FiberContent`
  - `SugarContent`
  - `ProteinContent`
- `RecipeInstructions`: Step-by-step cooking instructions

## Features

- **Nutritional Profile Matching**: Find recipes with similar nutritional characteristics
- **Customizable Thresholds**: Set maximum values for nutritional content to filter results
- **Ingredient Filtering**: Search for recipes containing specific ingredients
- **Flexible Pipeline**: Modular design allows for easy customization of the recommendation algorithm

## Technical Implementation

The recommendation system uses:

- **K-Nearest Neighbors (KNN)**: Core algorithm for finding similar recipes
- **Cosine Similarity**: Metric used to measure recipe similarity
- **StandardScaler**: Normalizes nutritional features for better comparison
- **Scikit-learn Pipeline**: Streamlines the data processing and recommendation workflow

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (for analysis and visualization)
- TensorFlow (for potential future advanced models)

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

## Usage

### Basic Recipe Recommendation

```python
# Load the dataset
dataset = pd.read_csv('recipes.csv')

# Define nutritional maximums
max_Calories = 2000
max_daily_fat = 100
max_daily_Saturatedfat = 13
max_daily_Cholesterol = 300
max_daily_Sodium = 2300
max_daily_Carbohydrate = 325
max_daily_Fiber = 40
max_daily_Sugar = 40
max_daily_Protein = 200
max_list = [max_Calories, max_daily_fat, max_daily_Saturatedfat, max_daily_Cholesterol, 
           max_daily_Sodium, max_daily_Carbohydrate, max_daily_Fiber, max_daily_Sugar, max_daily_Protein]

# Input nutritional values to match (Calories, Fat, SatFat, Cholesterol, Sodium, Carbs, Fiber, Sugar, Protein)
reference_nutrition = np.array([[350, 20, 5, 70, 600, 12, 3, 5, 35]])

# Get recommendations
recommendations = recommand(dataset, reference_nutrition, max_list)
```

### Finding Recipes with Specific Ingredients

```python
# Search for recipes containing eggs
egg_recipes = recommand(dataset, reference_nutrition, max_list, ingredient_filter=["egg"])

# Multiple ingredient filtering
vegetarian_recipes = recommand(dataset, reference_nutrition, max_list, 
                              ingredient_filter=["tofu", "vegetable"])
```

## How It Works

1. **Data Extraction**: The system first filters recipes based on nutritional constraints and optional ingredient requirements
2. **Feature Scaling**: Nutritional values are standardized to ensure fair comparisons
3. **KNN Modeling**: The K-Nearest Neighbors algorithm finds recipes with similar nutritional profiles
4. **Recommendation**: The system returns the most similar recipes to the reference input

## Function Reference

- `extract_data()`: Filters recipes based on nutritional maximums and ingredient requirements
- `scaling()`: Standardizes nutritional features for comparison
- `nn_predictor()`: Creates and trains the KNN model
- `build_pipeline()`: Constructs the recommendation pipeline
- `recommand()`: Main function that orchestrates the recommendation process

## Future Improvements

- Implement recipe rating prediction based on user preferences
- Add support for dietary restrictions (gluten-free, vegan, etc.)
- Develop a web interface for easier access to recommendations
- Incorporate flavor profile analysis for more nuanced recommendations
