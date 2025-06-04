"""
Personal Finance Analysis - Interactive Dashboard Module

This module creates comprehensive interactive dashboards for financial analysis visualization
and machine learning model insights presentation. Provides executive-level reporting
and stakeholder communication capabilities.

Author: Tyler Nguyen
Created: June 2025
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, timedelta
import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class FinancialAnalyticsDashboard:
    """
    Comprehensive interactive dashboard for financial data analysis and ML insights.
    
    This class provides enterprise-level dashboard capabilities including data visualization,
    model performance analysis, spending pattern insights, and future forecast presentation
    suitable for executive reporting and stakeholder communication.
    """
    
    def __init__(self, data_source='cleaned_financial_data.csv', 
                 model_artifacts='trained_expense_model.pkl'):
        """
        Initialize dashboard with data sources and model artifacts.
        
        Args:
            data_source (str): Path to processed financial dataset
            model_artifacts (str): Path to trained machine learning model
        """
        self.data_source = data_source
        self.model_artifacts_path = model_artifacts
        self.financial_dataset = None
        self.model_artifacts = None
        self.expense_transactions = None
        self.monthly_spending_series = None
        self.dashboard_components = {}
        
        logger.info(f"Initialized FinancialAnalyticsDashboard")
        logger.info(f"Data source: {data_source}")
        logger.info(f"Model artifacts: {model_artifacts}")
        
    def load_dashboard_data(self):
        """
        Load financial data and machine learning model artifacts for dashboard creation.
        
        Returns:
            bool: True if all required data loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If required data files are not accessible
            pickle.UnpicklingError: If model artifacts are corrupted
        """
        logger.info("Loading data sources for dashboard generation")
        
        try:
            # Load processed financial dataset
            self.financial_dataset = pd.read_csv(self.data_source)
            self.financial_dataset['Date'] = pd.to_datetime(self.financial_dataset['Date'])
            
            # Load trained machine learning model artifacts
            with open(self.model_artifacts_path, 'rb') as artifacts_file:
                self.model_artifacts = pickle.load(artifacts_file)
            
            # Prepare expense-focused analysis data
            self.expense_transactions = self.financial_dataset[
                self.financial_dataset['Transaction_Type'] == 'Expense'
            ].copy()
            
            # Generate monthly spending time series
            self.monthly_spending_series = self.expense_transactions.groupby(
                self.expense_transactions['Date'].dt.to_period('M')
            )['Amount'].sum().abs()
            
            dataset_size = len(self.financial_dataset)
            expense_count = len(self.expense_transactions)
            analysis_months = len(self.monthly_spending_series)
            
            logger.info(f"Dashboard data loaded successfully")
            logger.info(f"Total transactions: {dataset_size:,}")
            logger.info(f"Expense transactions: {expense_count:,}")
            logger.info(f"Analysis periods: {analysis_months} months")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Required data files not found: {str(e)}")
            logger.error("Please ensure data processing and ML training have been completed")
            return False
            
        except Exception as e:
            logger.error(f"Error loading dashboard data: {str(e)}")
            return False
    
    def analyze_spending_categories(self):
        """
        Perform comprehensive analysis of spending patterns by category.
        
        Analyzes transaction categories to identify spending distribution, frequency patterns,
        and average transaction values for business insight generation.
        
        Returns:
            pd.DataFrame: Category analysis results or None if insufficient data
        """
        logger.info("Analyzing spending patterns by transaction category")
        
        # Filter to categorized expense transactions
        categorized_expenses = self.expense_transactions[
            self.expense_transactions['Category'].notna()
        ]
        
        if len(categorized_expenses) == 0:
            logger.warning("No categorized transactions available for analysis")
            return None
        
        # Calculate comprehensive category metrics
        category_totals = categorized_expenses.groupby('Category')['Amount'].sum().abs()
        category_frequencies = categorized_expenses.groupby('Category').size()
        category_averages = (category_totals / category_frequencies).round(2)
        
        # Combine metrics into analysis dataframe
        category_analysis = pd.DataFrame({
            'Total_Spending': category_totals,
            'Transaction_Frequency': category_frequencies,
            'Average_Transaction_Value': category_averages,
            'Spending_Percentage': (category_totals / category_totals.sum() * 100).round(1)
        }).sort_values('Total_Spending', ascending=False)
        
        logger.info(f"Category analysis completed for {len(category_analysis)} categories")
        logger.info(f"Top category: {category_analysis.index[0]} (${category_analysis.iloc[0]['Total_Spending']:.0f})")
        
        return category_analysis
    
    def detect_spending_anomalies(self):
        """
        Identify and analyze anomalous spending patterns using statistical methods.
        
        Applies statistical outlier detection to identify periods of unusual spending
        behavior, such as travel months or major purchase periods, for enhanced
        financial planning and analysis.
        
        Returns:
            dict: Anomaly analysis results including thresholds and detected periods
        """
        logger.info("Performing statistical anomaly detection on spending patterns")
        
        # Calculate statistical parameters for anomaly detection
        mean_monthly_spending = self.monthly_spending_series.mean()
        std_monthly_spending = self.monthly_spending_series.std()
        
        # Define anomaly threshold using two standard deviations
        anomaly_threshold = mean_monthly_spending + 2 * std_monthly_spending
        
        # Identify anomalous and normal spending periods
        anomalous_periods = self.monthly_spending_series[
            self.monthly_spending_series > anomaly_threshold
        ]
        normal_periods = self.monthly_spending_series[
            self.monthly_spending_series <= anomaly_threshold
        ]
        
        anomaly_results = {
            'detection_threshold': anomaly_threshold,
            'anomalous_periods': anomalous_periods,
            'normal_periods': normal_periods,
            'anomaly_count': len(anomalous_periods),
            'normal_period_average': normal_periods.mean(),
            'anomalous_period_average': anomalous_periods.mean() if len(anomalous_periods) > 0 else 0
        }
        
        logger.info(f"Anomaly detection completed")
        logger.info(f"Detection threshold: ${anomaly_threshold:.0f}")
        logger.info(f"Anomalous periods identified: {len(anomalous_periods)}")
        
        return anomaly_results
    
    def create_comprehensive_dashboard(self):
        """
        Generate comprehensive interactive dashboard with multiple analytical views.
        
        Creates a multi-panel dashboard featuring spending trends, category analysis,
        model performance metrics, anomaly detection, and future forecasting for
        comprehensive financial analysis presentation.
        
        Returns:
            plotly.graph_objects.Figure: Complete interactive dashboard figure
        """
        logger.info("Creating comprehensive interactive financial analysis dashboard")
        
        # Retrieve analysis components
        category_analysis = self.analyze_spending_categories()
        anomaly_analysis = self.detect_spending_anomalies()
        
        # Initialize comprehensive subplot configuration
        dashboard_figure = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Monthly Spending Trend with ML Predictions',
                'Spending Distribution by Category',
                'Machine Learning Model Performance',
                'Spending Pattern Analysis by Day Type',
                'Normal vs Anomalous Spending Distribution',
                'Future Spending Forecast Comparison'
            ],
            specs=[
                [{"secondary_y": False}, {"type": "pie"}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Panel 1: Monthly Spending Trend with ML Predictions
        monthly_dates = [str(period) for period in self.monthly_spending_series.index]
        
        dashboard_figure.add_trace(
            go.Scatter(
                x=monthly_dates,
                y=self.monthly_spending_series.values,
                mode='lines+markers',
                name='Historical Spending',
                line=dict(color='#2E86C1', width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Monthly Spending: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add future predictions if available in model artifacts
        if 'future_predictions' in self.model_artifacts:
            last_period = self.monthly_spending_series.index[-1]
            future_periods = [str(last_period + i) for i in range(1, 4)]
            predictions = self.model_artifacts['future_predictions']
            
            dashboard_figure.add_trace(
                go.Scatter(
                    x=future_periods,
                    y=predictions,
                    mode='lines+markers',
                    name='ML Predictions',
                    line=dict(color='#E74C3C', width=3, dash='dash'),
                    marker=dict(size=8),
                    hovertemplate='<b>%{x}</b><br>Predicted Spending: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Highlight anomalous spending periods
        for period, amount in anomaly_analysis['anomalous_periods'].items():
            dashboard_figure.add_trace(
                go.Scatter(
                    x=[str(period)],
                    y=[amount],
                    mode='markers',
                    name='Anomalous Period',
                    marker=dict(size=15, color='#F39C12', symbol='star'),
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>Anomalous Spending: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Panel 2: Category Spending Distribution
        if category_analysis is not None:
            top_categories = category_analysis.head(6)
            
            dashboard_figure.add_trace(
                go.Pie(
                    labels=top_categories.index,
                    values=top_categories['Total_Spending'],
                    name="Category Distribution",
                    textinfo='label+percent',
                    textposition='auto',
                    hovertemplate='<b>%{label}</b><br>Total: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>',
                    marker=dict(colors=px.colors.qualitative.Set3)
                ),
                row=1, col=2
            )
        
        # Panel 3: ML Model Performance Comparison
        if 'all_trained_models' in self.model_artifacts:
            models = self.model_artifacts['all_trained_models']
            model_names = list(models.keys())
            performance_scores = [models[name]['test_mae'] for name in model_names]
            optimal_model = self.model_artifacts['optimal_model_name']
            
            bar_colors = ['#27AE60' if name == optimal_model else '#E74C3C' for name in model_names]
            
            dashboard_figure.add_trace(
                go.Bar(
                    x=model_names,
                    y=performance_scores,
                    name='Model Performance',
                    marker_color=bar_colors,
                    text=[f'${score:.0f}' for score in performance_scores],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Mean Absolute Error: $%{y:.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Panel 4: Spending Pattern Analysis by Day Type
        weekend_spending = self.expense_transactions[
            self.expense_transactions['Is_Weekend'] == True
        ]['Amount'].abs()
        weekday_spending = self.expense_transactions[
            self.expense_transactions['Is_Weekend'] == False
        ]['Amount'].abs()
        
        dashboard_figure.add_trace(
            go.Box(
                y=weekend_spending,
                name='Weekend Spending',
                marker_color='#F39C12',
                boxpoints='outliers',
                hovertemplate='Weekend Transaction: $%{y:.0f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        dashboard_figure.add_trace(
            go.Box(
                y=weekday_spending,
                name='Weekday Spending',
                marker_color='#2E86C1',
                boxpoints='outliers',
                hovertemplate='Weekday Transaction: $%{y:.0f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Panel 5: Normal vs Anomalous Spending Distribution
        dashboard_figure.add_trace(
            go.Histogram(
                x=anomaly_analysis['normal_periods'],
                name='Normal Spending Months',
                nbinsx=8,
                opacity=0.7,
                marker_color='#2E86C1',
                hovertemplate='Spending Range: $%{x}<br>Frequency: %{y}<extra></extra>'
            ),
            row=3, col=1
        )
        
        if len(anomaly_analysis['anomalous_periods']) > 0:
            dashboard_figure.add_trace(
                go.Histogram(
                    x=anomaly_analysis['anomalous_periods'],
                    name='Anomalous Spending Months',
                    nbinsx=3,
                    opacity=0.7,
                    marker_color='#F39C12',
                    hovertemplate='Spending Range: $%{x}<br>Frequency: %{y}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Panel 6: Future Forecast Comparison
        recent_average = self.monthly_spending_series.tail(3).mean()
        normal_average = anomaly_analysis['normal_period_average']
        
        forecast_comparison = {
            'Recent 3-Month Average': recent_average,
            'Historical Normal Average': normal_average
        }
        
        if 'future_predictions' in self.model_artifacts:
            ml_prediction_average = np.mean(self.model_artifacts['future_predictions'])
            forecast_comparison['ML Prediction Average'] = ml_prediction_average
        
        comparison_categories = list(forecast_comparison.keys())
        comparison_values = list(forecast_comparison.values())
        comparison_colors = ['#2E86C1', '#27AE60', '#E74C3C'][:len(comparison_categories)]
        
        dashboard_figure.add_trace(
            go.Bar(
                x=comparison_categories,
                y=comparison_values,
                name='Forecast Comparison',
                marker_color=comparison_colors,
                text=[f'${val:.0f}' for val in comparison_values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Average: $%{y:,.0f}<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Configure dashboard layout and styling
        dashboard_figure.update_layout(
            title={
                'text': 'Personal Finance Analytics Dashboard<br><sub>Comprehensive Spending Analysis & Machine Learning Insights</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            height=1000,
            showlegend=True,
            template='plotly_white',
            hovermode='closest',
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        # Update axis labels and formatting
        dashboard_figure.update_xaxes(title_text="Time Period", row=1, col=1)
        dashboard_figure.update_yaxes(title_text="Monthly Spending ($)", row=1, col=1)
        dashboard_figure.update_xaxes(title_text="ML Model", row=2, col=1)
        dashboard_figure.update_yaxes(title_text="Mean Absolute Error ($)", row=2, col=1)
        dashboard_figure.update_yaxes(title_text="Transaction Amount ($)", row=2, col=2)
        dashboard_figure.update_xaxes(title_text="Monthly Spending ($)", row=3, col=1)
        dashboard_figure.update_yaxes(title_text="Frequency", row=3, col=1)
        dashboard_figure.update_xaxes(title_text="Forecast Type", row=3, col=2)
        dashboard_figure.update_yaxes(title_text="Average Amount ($)", row=3, col=2)
        
        logger.info("Comprehensive dashboard created successfully")
        return dashboard_figure
    
    def generate_executive_insights(self):
        """
        Generate comprehensive business insights and key performance indicators.
        
        Analyzes financial data to extract actionable business intelligence including
        spending patterns, model performance metrics, and strategic recommendations
        for executive-level reporting.
        
        Returns:
            dict: Comprehensive insights and KPI metrics
        """
        logger.info("Generating executive insights and key performance indicators")
        
        category_analysis = self.analyze_spending_categories()
        anomaly_analysis = self.detect_spending_anomalies()
        
        # Calculate comprehensive business metrics
        executive_insights = {
            'total_transactions': len(self.financial_dataset),
            'analysis_period_days': (
                self.financial_dataset['Date'].max() - self.financial_dataset['Date'].min()
            ).days,
            'total_expense_amount': self.expense_transactions['Amount'].abs().sum(),
            'average_monthly_spending': self.monthly_spending_series.mean(),
            'maximum_monthly_spending': self.monthly_spending_series.max(),
            'minimum_monthly_spending': self.monthly_spending_series.min(),
            'spending_volatility': self.monthly_spending_series.std(),
            'anomalous_period_count': anomaly_analysis['anomaly_count'],
            'normal_period_count': len(anomaly_analysis['normal_periods']),
            'spending_efficiency_ratio': (
                anomaly_analysis['normal_period_average'] / self.monthly_spending_series.mean()
            )
        }
        
        # Add category-specific insights
        if category_analysis is not None:
            executive_insights.update({
                'primary_spending_category': category_analysis.index[0],
                'primary_category_amount': category_analysis.iloc[0]['Total_Spending'],
                'category_diversification_index': len(category_analysis),
                'top_category_concentration': category_analysis.iloc[0]['Spending_Percentage']
            })
        
        # Add machine learning performance metrics
        if self.model_artifacts and 'optimal_model_name' in self.model_artifacts:
            optimal_performance = self.model_artifacts['all_trained_models'][
                self.model_artifacts['optimal_model_name']
            ]
            executive_insights.update({
                'ml_prediction_accuracy': optimal_performance['test_mae'],
                'ml_model_confidence': optimal_performance['test_r2'],
                'optimal_algorithm': self.model_artifacts['optimal_model_name']
            })
        
        logger.info("Executive insights generated successfully")
        return executive_insights
    
    def export_dashboard_report(self, output_filename='financial_analytics_dashboard.html'):
        """
        Export interactive dashboard as standalone HTML report.
        
        Generates a self-contained HTML file suitable for sharing with stakeholders,
        embedding in presentations, or hosting on web platforms.
        
        Args:
            output_filename (str): Output filename for HTML dashboard export
            
        Returns:
            str: Path to exported dashboard file
        """
        logger.info(f"Exporting interactive dashboard to {output_filename}")
        
        dashboard_figure = self.create_comprehensive_dashboard()
        dashboard_figure.write_html(
            output_filename, 
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
        logger.info(f"Dashboard exported successfully to {output_filename}")
        return output_filename
    
    def display_interactive_dashboard(self):
        """
        Display interactive dashboard in default web browser.
        
        Opens the comprehensive dashboard in the default web browser for
        immediate analysis and exploration.
        
        Returns:
            plotly.graph_objects.Figure: Dashboard figure object
        """
        logger.info("Displaying interactive dashboard in web browser")
        
        dashboard_figure = self.create_comprehensive_dashboard()
        dashboard_figure.show()
        
        return dashboard_figure
    
    def print_executive_summary(self):
        """
        Print comprehensive executive summary report to console.
        
        Generates a detailed text-based summary suitable for logging,
        documentation, or stakeholder communication.
        """
        logger.info("Generating executive summary report")
        
        insights = self.generate_executive_insights()
        
        print("\n" + "="*100)
        print("PERSONAL FINANCE ANALYTICS - EXECUTIVE SUMMARY REPORT")
        print("="*100)
        
        print(f"\nDataset Overview:")
        print(f"  Total Transactions Analyzed: {insights['total_transactions']:,}")
        print(f"  Analysis Time Period: {insights['analysis_period_days']} days")
        print(f"  Total Expense Amount: ${insights['total_expense_amount']:,.2f}")
        
        print(f"\nSpending Pattern Analysis:")
        print(f"  Average Monthly Spending: ${insights['average_monthly_spending']:,.2f}")
        print(f"  Monthly Spending Range: ${insights['minimum_monthly_spending']:,.2f} - ${insights['maximum_monthly_spending']:,.2f}")
        print(f"  Spending Volatility (Std Dev): ${insights['spending_volatility']:,.2f}")
        
        if 'primary_spending_category' in insights:
            print(f"\nCategory Distribution Analysis:")
            print(f"  Primary Spending Category: {insights['primary_spending_category']}")
            print(f"  Primary Category Amount: ${insights['primary_category_amount']:,.2f}")
            print(f"  Category Concentration: {insights['top_category_concentration']:.1f}%")
            print(f"  Spending Diversification: {insights['category_diversification_index']} categories")
        
        if 'ml_prediction_accuracy' in insights:
            print(f"\nMachine Learning Model Performance:")
            print(f"  Optimal Algorithm: {insights['optimal_algorithm']}")
            print(f"  Prediction Accuracy: ±${insights['ml_prediction_accuracy']:,.0f}")
            print(f"  Model Confidence (R²): {insights['ml_model_confidence']:.1%}")
        
        print(f"\nAnomaly Detection Results:")
        print(f"  Normal Spending Periods: {insights['normal_period_count']}")
        print(f"  Anomalous Spending Periods: {insights['anomalous_period_count']}")
        print(f"  Spending Efficiency Ratio: {insights['spending_efficiency_ratio']:.2f}")
        
        logger.info("Executive summary report generated successfully")


def execute_dashboard_pipeline():
    """
    Execute comprehensive dashboard creation and reporting pipeline.
    
    Orchestrates the complete dashboard workflow including data loading,
    analysis computation, visualization creation, and report generation
    for comprehensive financial analytics presentation.
    
    Returns:
        FinancialAnalyticsDashboard: Dashboard instance or None if pipeline fails
    """
    logger.info("="*100)
    logger.info("PERSONAL FINANCE INTERACTIVE DASHBOARD PIPELINE EXECUTION")
    logger.info("="*100)
    
    # Initialize dashboard pipeline
    analytics_dashboard = FinancialAnalyticsDashboard()
    
    try:
        # Execute dashboard creation pipeline
        if not analytics_dashboard.load_dashboard_data():
            return None
        
        # Generate and display comprehensive dashboard
        analytics_dashboard.display_interactive_dashboard()
        
        # Generate executive summary report
        analytics_dashboard.print_executive_summary()
        
        # Export dashboard for sharing and presentation
        analytics_dashboard.export_dashboard_report()
        
        logger.info("Dashboard pipeline execution completed successfully")
        logger.info("Interactive dashboard created and exported for stakeholder use")
        
        return analytics_dashboard
        
    except Exception as e:
        logger.error(f"Dashboard pipeline execution failed: {str(e)}")
        logger.error("Please verify data processing and ML model training completion")
        return None


if __name__ == "__main__":
    # Execute dashboard pipeline when script is run directly
    dashboard_instance = execute_dashboard_pipeline()