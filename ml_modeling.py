"""
Personal Finance Analysis - Machine Learning Module

This module implements machine learning models for financial expense prediction and analysis.
Provides comprehensive model training, evaluation, and prediction capabilities for budget forecasting.

Author: Tyler Nguyen
Created: June 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import logging
import warnings

# Configure logging for production environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')


class ExpensePredictionModel:
    """
    Machine learning pipeline for financial expense prediction and analysis.
    
    This class encapsulates the complete ML workflow including data preparation,
    feature engineering, model training, evaluation, and prediction capabilities
    for personal finance analysis.
    """
    
    def __init__(self, data_source='cleaned_financial_data.csv'):
        """
        Initialize the machine learning model with data source configuration.
        
        Args:
            data_source (str): Path to cleaned financial dataset CSV file
        """
        self.data_source = data_source
        self.dataset = None
        self.feature_matrix = None
        self.target_vector = None
        self.trained_models = {}
        self.feature_scaler = StandardScaler()
        self.optimal_model = None
        self.optimal_model_name = None
        self.model_performance_metrics = {}
        
        logger.info(f"Initialized ExpensePredictionModel with data source: {data_source}")
        
    def load_financial_data(self):
        """
        Load preprocessed financial data for machine learning model training.
        
        Returns:
            bool: True if data loading successful, False otherwise
            
        Raises:
            FileNotFoundError: If the specified data file does not exist
            pd.errors.EmptyDataError: If the data file is empty or corrupted
        """
        logger.info("Loading financial data for machine learning pipeline")
        
        try:
            self.dataset = pd.read_csv(self.data_source)
            self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
            
            record_count = len(self.dataset)
            date_range = self.dataset['Date'].max() - self.dataset['Date'].min()
            
            logger.info(f"Successfully loaded {record_count:,} financial records")
            logger.info(f"Dataset spans {date_range.days} days")
            
            return True
            
        except FileNotFoundError:
            logger.error(f"Financial data file not found: {self.data_source}")
            logger.error("Please ensure data processing pipeline has been executed")
            return False
            
        except Exception as e:
            logger.error(f"Error loading financial data: {str(e)}")
            return False
    
    def engineer_ml_features(self):
        """
        Create machine learning features from raw transaction data.
        
        Transforms transaction-level data into monthly aggregated features suitable
        for time series prediction modeling. Includes temporal features, spending
        patterns, and lagged variables for enhanced predictive power.
        
        Returns:
            pd.DataFrame: Engineered feature matrix for model training
            
        Raises:
            ValueError: If insufficient data for feature engineering
        """
        logger.info("Starting feature engineering for machine learning models")
        
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_financial_data() first")
        
        # Filter to expense transactions for spending prediction
        expense_data = self.dataset[self.dataset['Transaction_Type'] == 'Expense'].copy()
        
        if len(expense_data) == 0:
            raise ValueError("No expense transactions found in dataset")
        
        # Create monthly period grouping for aggregation
        expense_data['Period'] = expense_data['Date'].dt.to_period('M')
        
        # Aggregate transaction data by month for time series modeling
        monthly_aggregations = expense_data.groupby('Period').agg({
            'Amount': ['sum', 'count', 'mean', 'std'],
            'Is_Weekend': 'sum'
        }).round(2)
        
        # Flatten multi-level column index
        monthly_aggregations.columns = [
            'Total_Monthly_Spending', 'Transaction_Count', 'Average_Transaction_Amount', 
            'Spending_Volatility', 'Weekend_Transaction_Count'
        ]
        
        # Convert spending to positive values for interpretability
        monthly_aggregations['Total_Monthly_Spending'] = monthly_aggregations['Total_Monthly_Spending'].abs()
        
        # Engineer temporal features for seasonality modeling
        monthly_aggregations['Month_Number'] = monthly_aggregations.index.month
        monthly_aggregations['Quarter'] = monthly_aggregations.index.quarter
        monthly_aggregations['Days_In_Month'] = monthly_aggregations.index.days_in_month
        
        # Create lagged features for time series prediction
        monthly_aggregations['Previous_Month_Spending'] = monthly_aggregations['Total_Monthly_Spending'].shift(1)
        monthly_aggregations['Spending_Momentum'] = (
            monthly_aggregations['Total_Monthly_Spending'] - monthly_aggregations['Previous_Month_Spending']
        )
        monthly_aggregations['Spending_Growth_Rate'] = (
            monthly_aggregations['Spending_Momentum'] / monthly_aggregations['Previous_Month_Spending'] * 100
        ).round(2)
        
        # Identify anomalous spending periods for feature creation
        spending_mean = monthly_aggregations['Total_Monthly_Spending'].mean()
        spending_std = monthly_aggregations['Total_Monthly_Spending'].std()
        anomaly_threshold = spending_mean + 2 * spending_std
        
        monthly_aggregations['Is_Anomalous_Spending'] = (
            monthly_aggregations['Total_Monthly_Spending'] > anomaly_threshold
        ).astype(int)
        
        # Remove initial periods without lagged features
        self.feature_matrix = monthly_aggregations.dropna()
        
        feature_count = len(self.feature_matrix)
        feature_names = list(self.feature_matrix.columns)
        anomaly_count = self.feature_matrix['Is_Anomalous_Spending'].sum()
        
        logger.info(f"Feature engineering completed successfully")
        logger.info(f"Generated {feature_count} monthly feature records")
        logger.info(f"Created {len(feature_names)} predictive features")
        logger.info(f"Identified {anomaly_count} anomalous spending periods")
        
        return self.feature_matrix
    
    def prepare_training_data(self, test_size=0.3, random_state=42):
        """
        Prepare feature matrix and target vector for model training.
        
        Selects relevant predictive features and creates training/testing splits
        with proper temporal ordering for time series validation.
        
        Args:
            test_size (float): Proportion of data reserved for testing
            random_state (int): Random seed for reproducible splits
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) training and testing datasets
        """
        logger.info("Preparing training data for machine learning models")
        
        # Define feature columns for prediction model
        predictive_features = [
            'Transaction_Count', 'Average_Transaction_Amount', 'Weekend_Transaction_Count',
            'Month_Number', 'Quarter', 'Days_In_Month', 'Previous_Month_Spending',
            'Is_Anomalous_Spending'
        ]
        
        # Create feature matrix and target vector
        X = self.feature_matrix[predictive_features]
        y = self.feature_matrix['Total_Monthly_Spending']
        
        # Split data maintaining temporal order (no shuffle for time series)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=random_state
        )
        
        logger.info(f"Training data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        logger.info(f"Feature dimensions: {X_train.shape[1]} predictive features")
        
        return X_train, X_test, y_train, y_test
    
    def train_prediction_models(self, test_size=0.3):
        """
        Train multiple machine learning models and evaluate performance.
        
        Implements comprehensive model comparison including linear regression,
        regularized regression, and ensemble methods with proper evaluation metrics.
        
        Args:
            test_size (float): Proportion of data for model testing
            
        Returns:
            dict: Model performance results and trained model objects
        """
        logger.info("Initiating machine learning model training pipeline")
        
        # Prepare training and testing datasets
        X_train, X_test, y_train, y_test = self.prepare_training_data(test_size)
        
        # Apply feature scaling for algorithm optimization
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Define model configurations for comparison
        model_configurations = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0, random_state=42),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        model_results = {}
        
        # Train and evaluate each model configuration
        for model_name, model_instance in model_configurations.items():
            logger.info(f"Training {model_name} model")
            
            # Train model on scaled training data
            model_instance.fit(X_train_scaled, y_train)
            
            # Generate predictions for evaluation
            train_predictions = model_instance.predict(X_train_scaled)
            test_predictions = model_instance.predict(X_test_scaled)
            
            # Calculate comprehensive evaluation metrics
            train_mae = mean_absolute_error(y_train, train_predictions)
            test_mae = mean_absolute_error(y_test, test_predictions)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            train_r2 = r2_score(y_train, train_predictions)
            test_r2 = r2_score(y_test, test_predictions)
            
            # Store comprehensive model results
            model_results[model_name] = {
                'model_instance': model_instance,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_actuals': y_test,
                'test_predictions': test_predictions
            }
            
            logger.info(f"{model_name} performance - MAE: ${test_mae:.0f}, R²: {test_r2:.3f}")
        
        # Identify optimal model based on test MAE
        self.optimal_model_name = min(model_results.items(), key=lambda x: x[1]['test_mae'])[0]
        self.optimal_model = model_results[self.optimal_model_name]['model_instance']
        self.trained_models = model_results
        
        optimal_mae = model_results[self.optimal_model_name]['test_mae']
        optimal_r2 = model_results[self.optimal_model_name]['test_r2']
        
        logger.info(f"Model training completed successfully")
        logger.info(f"Optimal model: {self.optimal_model_name}")
        logger.info(f"Best performance: MAE=${optimal_mae:.0f}, R²={optimal_r2:.3f}")
        
        return model_results
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance for model interpretability.
        
        Extracts and ranks feature importance scores from tree-based models
        to understand which variables most strongly influence spending predictions.
        
        Returns:
            pd.DataFrame: Feature importance rankings or None if unavailable
        """
        logger.info("Analyzing feature importance for model interpretability")
        
        if 'Random_Forest' not in self.trained_models:
            logger.warning("Random Forest model not available for feature importance analysis")
            return None
        
        rf_model = self.trained_models['Random_Forest']['model_instance']
        feature_names = [
            'Transaction_Count', 'Average_Transaction_Amount', 'Weekend_Transaction_Count',
            'Month_Number', 'Quarter', 'Days_In_Month', 'Previous_Month_Spending',
            'Is_Anomalous_Spending'
        ]
        
        # Create feature importance dataframe
        importance_analysis = pd.DataFrame({
            'Feature_Name': feature_names,
            'Importance_Score': rf_model.feature_importances_
        }).sort_values('Importance_Score', ascending=False)
        
        logger.info("Feature importance analysis completed")
        for _, row in importance_analysis.head(5).iterrows():
            logger.info(f"  {row['Feature_Name']}: {row['Importance_Score']:.3f}")
        
        return importance_analysis
    
    def generate_spending_forecasts(self, forecast_horizon=3):
        """
        Generate future spending predictions using the optimal trained model.
        
        Creates forward-looking spending forecasts based on historical patterns
        and trained model parameters. Assumes normal spending conditions without
        anomalous events.
        
        Args:
            forecast_horizon (int): Number of months to forecast forward
            
        Returns:
            list: Predicted monthly spending amounts for forecast period
        """
        logger.info(f"Generating {forecast_horizon}-month spending forecasts")
        
        if self.optimal_model is None:
            logger.error("No trained model available for forecasting")
            return None
        
        # Extract most recent period data for forecasting base
        latest_period = self.feature_matrix.index.max()
        latest_features = self.feature_matrix.loc[latest_period].copy()
        latest_spending = self.feature_matrix.loc[latest_period, 'Total_Monthly_Spending']
        
        forecast_predictions = []
        
        # Generate sequential predictions for each forecast period
        for period_offset in range(1, forecast_horizon + 1):
            future_period = latest_period + period_offset
            forecast_features = latest_features.copy()
            
            # Update temporal features for future period
            forecast_features['Month_Number'] = future_period.month
            forecast_features['Quarter'] = future_period.quarter
            forecast_features['Days_In_Month'] = future_period.days_in_month
            forecast_features['Previous_Month_Spending'] = (
                latest_spending if period_offset == 1 else forecast_predictions[-1]
            )
            forecast_features['Is_Anomalous_Spending'] = 0  # Assume normal spending
            
            # Select features used in model training
            model_features = [
                'Transaction_Count', 'Average_Transaction_Amount', 'Weekend_Transaction_Count',
                'Month_Number', 'Quarter', 'Days_In_Month', 'Previous_Month_Spending',
                'Is_Anomalous_Spending'
            ]
            
            # Generate scaled prediction
            feature_vector = self.feature_scaler.transform([forecast_features[model_features]])
            period_prediction = self.optimal_model.predict(feature_vector)[0]
            
            forecast_predictions.append(period_prediction)
            logger.info(f"  {future_period}: ${period_prediction:.0f}")
        
        logger.info("Spending forecasts generated successfully")
        return forecast_predictions
    
    def create_model_visualizations(self):
        """
        Generate comprehensive visualizations for model analysis and validation.
        
        Creates multiple plots including model performance comparison, actual vs predicted
        analysis, feature importance rankings, and spending trend visualization.
        
        Returns:
            matplotlib.figure.Figure: Comprehensive model analysis figure
        """
        logger.info("Creating model performance and analysis visualizations")
        
        # Initialize subplot configuration
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Machine Learning Model Analysis - Personal Finance Prediction', fontsize=16)
        
        # Plot 1: Model Performance Comparison
        model_names = list(self.trained_models.keys())
        test_mae_scores = [self.trained_models[name]['test_mae'] for name in model_names]
        
        bar_colors = ['green' if name == self.optimal_model_name else 'lightblue' for name in model_names]
        axes[0, 0].bar(model_names, test_mae_scores, color=bar_colors)
        axes[0, 0].set_title('Model Performance Comparison (Lower is Better)')
        axes[0, 0].set_ylabel('Test Mean Absolute Error ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Actual vs Predicted Analysis
        optimal_results = self.trained_models[self.optimal_model_name]
        axes[0, 1].scatter(optimal_results['test_actuals'], optimal_results['test_predictions'], alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(optimal_results['test_actuals'].min(), optimal_results['test_predictions'].min())
        max_val = max(optimal_results['test_actuals'].max(), optimal_results['test_predictions'].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        axes[0, 1].set_xlabel('Actual Monthly Spending ($)')
        axes[0, 1].set_ylabel('Predicted Monthly Spending ($)')
        axes[0, 1].set_title(f'{self.optimal_model_name}: Prediction Accuracy')
        
        # Plot 3: Monthly Spending Trend
        monthly_spending_values = self.feature_matrix['Total_Monthly_Spending']
        month_indices = range(len(monthly_spending_values))
        
        axes[1, 0].plot(month_indices, monthly_spending_values, 'b-o', markersize=4)
        axes[1, 0].set_title('Historical Monthly Spending Trend')
        axes[1, 0].set_xlabel('Month Index')
        axes[1, 0].set_ylabel('Monthly Spending ($)')
        
        # Plot 4: Feature Importance Analysis
        feature_importance = self.analyze_feature_importance()
        if feature_importance is not None:
            top_features = feature_importance.head(6)
            axes[1, 1].barh(top_features['Feature_Name'], top_features['Importance_Score'])
            axes[1, 1].set_title('Feature Importance Rankings')
            axes[1, 1].set_xlabel('Importance Score')
        
        plt.tight_layout()
        logger.info("Model visualization suite created successfully")
        
        return fig
    
    def save_trained_model(self, output_path='trained_expense_model.pkl'):
        """
        Serialize and save the complete trained model pipeline.
        
        Persists the trained model, preprocessing components, and metadata
        for future deployment and prediction use.
        
        Args:
            output_path (str): File path for saved model artifacts
        """
        logger.info(f"Saving trained model pipeline to {output_path}")
        
        model_artifacts = {
            'optimal_model': self.optimal_model,
            'optimal_model_name': self.optimal_model_name,
            'feature_scaler': self.feature_scaler,
            'all_trained_models': self.trained_models,
            'feature_columns': [
                'Transaction_Count', 'Average_Transaction_Amount', 'Weekend_Transaction_Count',
                'Month_Number', 'Quarter', 'Days_In_Month', 'Previous_Month_Spending',
                'Is_Anomalous_Spending'
            ],
            'model_metadata': {
                'training_date': pd.Timestamp.now(),
                'feature_count': len(self.feature_matrix),
                'optimal_model_performance': self.trained_models[self.optimal_model_name]['test_mae']
            }
        }
        
        with open(output_path, 'wb') as model_file:
            pickle.dump(model_artifacts, model_file)
        
        logger.info("Model pipeline saved successfully")
    
    def generate_comprehensive_report(self):
        """
        Generate detailed machine learning model performance and analysis report.
        
        Provides comprehensive summary of model training results, performance metrics,
        feature analysis, and business insights for stakeholder communication.
        """
        logger.info("Generating comprehensive machine learning model report")
        
        if not self.trained_models:
            logger.error("No trained models available for reporting")
            return
        
        print("\n" + "="*80)
        print("MACHINE LEARNING MODEL PERFORMANCE REPORT")
        print("="*80)
        
        # Model Performance Summary
        optimal_results = self.trained_models[self.optimal_model_name]
        print(f"\nModel Performance Summary:")
        print(f"  Optimal Algorithm: {self.optimal_model_name}")
        print(f"  Prediction Accuracy (MAE): ±${optimal_results['test_mae']:.0f}")
        print(f"  Variance Explained (R²): {optimal_results['test_r2']:.1%}")
        print(f"  Root Mean Square Error: ${optimal_results['test_rmse']:.0f}")
        
        # Dataset Analysis
        monthly_spending = self.feature_matrix['Total_Monthly_Spending']
        anomalous_periods = self.feature_matrix[self.feature_matrix['Is_Anomalous_Spending'] == 1]
        
        print(f"\nDataset Analysis:")
        print(f"  Training Periods: {len(self.feature_matrix)} months")
        print(f"  Average Monthly Spending: ${monthly_spending.mean():.0f}")
        print(f"  Spending Range: ${monthly_spending.min():.0f} - ${monthly_spending.max():.0f}")
        print(f"  Anomalous Periods Detected: {len(anomalous_periods)}")
        
        if len(anomalous_periods) > 0:
            print(f"  Average Anomalous Spending: ${anomalous_periods['Total_Monthly_Spending'].mean():.0f}")
        
        # Model Comparison Results
        print(f"\nAll Model Performance Results:")
        for model_name, results in self.trained_models.items():
            status_indicator = "  [OPTIMAL]" if model_name == self.optimal_model_name else "           "
            print(f"{status_indicator} {model_name}: MAE=${results['test_mae']:.0f}, R²={results['test_r2']:.3f}")
        
        # Feature Importance Insights
        feature_analysis = self.analyze_feature_importance()
        if feature_analysis is not None:
            print(f"\nMost Predictive Features:")
            for _, row in feature_analysis.head(3).iterrows():
                print(f"  {row['Feature_Name']}: {row['Importance_Score']:.3f}")
        
        logger.info("Comprehensive model report generated successfully")


def execute_ml_pipeline():
    """
    Execute the complete machine learning pipeline for expense prediction.
    
    Orchestrates the entire ML workflow from data loading through model training,
    evaluation, and artifact generation. Provides comprehensive error handling
    and logging for production deployment.
    
    Returns:
        ExpensePredictionModel: Trained model instance or None if pipeline fails
    """
    logger.info("="*80)
    logger.info("PERSONAL FINANCE MACHINE LEARNING PIPELINE EXECUTION")
    logger.info("="*80)
    
    # Initialize machine learning pipeline
    ml_pipeline = ExpensePredictionModel()
    
    try:
        # Execute pipeline stages sequentially
        if not ml_pipeline.load_financial_data():
            return None
        
        ml_pipeline.engineer_ml_features()
        ml_pipeline.train_prediction_models()
        ml_pipeline.analyze_feature_importance()
        
        # Generate future predictions
        future_forecasts = ml_pipeline.generate_spending_forecasts()
        
        # Create comprehensive visualizations
        ml_pipeline.create_model_visualizations()
        
        # Generate detailed performance report
        ml_pipeline.generate_comprehensive_report()
        
        # Save trained model artifacts
        ml_pipeline.save_trained_model()
        
        logger.info("Machine learning pipeline executed successfully")
        logger.info("Model ready for deployment and prediction use")
        
        return ml_pipeline
        
    except Exception as e:
        logger.error(f"Machine learning pipeline execution failed: {str(e)}")
        logger.error("Please verify data preprocessing and input requirements")
        return None


if __name__ == "__main__":
    # Execute machine learning pipeline when script is run directly
    trained_model_pipeline = execute_ml_pipeline()