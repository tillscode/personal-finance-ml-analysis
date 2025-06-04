"""
Personal Finance Analysis - Data Processing Module

This module handles data loading, cleaning, and preprocessing from multiple financial sources.
Implements standardized data pipeline for transaction analysis and machine learning preparation.

Author: Tyler Nguyen
Created: June 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def load_wells_fargo_data(filepath='wells_fargo_checking.csv'):
    """
    Load and standardize Wells Fargo checking account transaction data.
    
    Wells Fargo CSV files typically have no header row and require manual column assignment.
    This function handles the specific format and standardizes column names for downstream processing.
    
    Args:
        filepath (str): Path to Wells Fargo CSV export file
        
    Returns:
        pd.DataFrame: Standardized transaction data with consistent column structure
        
    Raises:
        FileNotFoundError: If the specified file path does not exist
        pd.errors.EmptyDataError: If the CSV file is empty or corrupted
    """
    try:
        # Load data without header as Wells Fargo exports don't include column names
        df = pd.read_csv(filepath, header=None)
        
        # Assign standardized column names based on Wells Fargo export format
        df.columns = ['Date', 'Amount', 'Status', 'Balance', 'Description']
        
        # Select only required columns for analysis
        df = df[['Date', 'Amount', 'Description']].copy()
        
        # Add source identifier for data lineage tracking
        df['Source'] = 'Wells_Fargo'
        
        logger.info(f"Successfully loaded {len(df)} transactions from Wells Fargo")
        return df
        
    except FileNotFoundError:
        logger.error(f"Wells Fargo data file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading Wells Fargo data: {str(e)}")
        raise


def load_discover_data(filepath='discover_cc.csv'):
    """
    Load and standardize Discover credit card transaction data.
    
    Discover CSV exports include headers and category information. Credit card transactions
    are converted to negative values to represent expenses consistently across data sources.
    
    Args:
        filepath (str): Path to Discover CSV export file
        
    Returns:
        pd.DataFrame: Standardized transaction data with expense amounts as negative values
        
    Raises:
        FileNotFoundError: If the specified file path does not exist
        KeyError: If expected columns are missing from the CSV file
    """
    try:
        # Load Discover data with headers
        df = pd.read_csv(filepath)
        
        # Select and rename columns to match standardized format
        required_columns = ['Trans. Date', 'Amount', 'Description', 'Category']
        df = df[required_columns].copy()
        df.columns = ['Date', 'Amount', 'Description', 'Category']
        
        # Convert credit card charges to negative values for expense tracking
        # Discover exports show purchases as positive amounts
        df['Amount'] = -df['Amount']
        
        # Add source identifier
        df['Source'] = 'Discover'
        
        logger.info(f"Successfully loaded {len(df)} transactions from Discover")
        return df
        
    except FileNotFoundError:
        logger.error(f"Discover data file not found: {filepath}")
        raise
    except KeyError as e:
        logger.error(f"Missing expected columns in Discover data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading Discover data: {str(e)}")
        raise


def load_chase_data(filepath='chase_cc.csv'):
    """
    Load and standardize Chase credit card transaction data.
    
    Chase CSV exports include headers and category classifications. Similar to Discover,
    transactions are converted to negative values for consistent expense representation.
    
    Args:
        filepath (str): Path to Chase CSV export file
        
    Returns:
        pd.DataFrame: Standardized transaction data with expense amounts as negative values
        
    Raises:
        FileNotFoundError: If the specified file path does not exist
        KeyError: If expected columns are missing from the CSV file
    """
    try:
        # Load Chase data with headers
        df = pd.read_csv(filepath)
        
        # Select and rename columns to match standardized format
        required_columns = ['Transaction Date', 'Amount', 'Description', 'Category']
        df = df[required_columns].copy()
        df.columns = ['Date', 'Amount', 'Description', 'Category']
        
        # Convert credit card charges to negative values for expense tracking
        df['Amount'] = -df['Amount']
        
        # Add source identifier
        df['Source'] = 'Chase'
        
        logger.info(f"Successfully loaded {len(df)} transactions from Chase")
        return df
        
    except FileNotFoundError:
        logger.error(f"Chase data file not found: {filepath}")
        raise
    except KeyError as e:
        logger.error(f"Missing expected columns in Chase data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading Chase data: {str(e)}")
        raise


def combine_financial_datasets(wells_path='wells_fargo_checking.csv',
                              discover_path='discover_cc.csv', 
                              chase_path='chase_cc.csv'):
    """
    Load and combine transaction data from multiple financial institutions.
    
    This function orchestrates the loading of data from three different sources,
    each with their own format requirements, and combines them into a unified dataset
    suitable for analysis and machine learning.
    
    Args:
        wells_path (str): File path to Wells Fargo checking account data
        discover_path (str): File path to Discover credit card data
        chase_path (str): File path to Chase credit card data
        
    Returns:
        pd.DataFrame: Combined transaction dataset with standardized schema
        
    Raises:
        Exception: If any data source fails to load or if combination fails
    """
    logger.info("Initiating multi-source financial data integration")
    
    try:
        # Load data from each financial institution
        wells_transactions = load_wells_fargo_data(wells_path)
        discover_transactions = load_discover_data(discover_path)
        chase_transactions = load_chase_data(chase_path)
        
        # Combine all datasets using pandas concat
        combined_dataset = pd.concat([
            wells_transactions, 
            discover_transactions, 
            chase_transactions
        ], ignore_index=True)
        
        # Log summary statistics for data integration validation
        total_transactions = len(combined_dataset)
        source_breakdown = combined_dataset['Source'].value_counts()
        
        logger.info(f"Data integration completed successfully")
        logger.info(f"Total transactions: {total_transactions:,}")
        for source, count in source_breakdown.items():
            logger.info(f"  {source}: {count:,} transactions")
        
        return combined_dataset
        
    except Exception as e:
        logger.error(f"Failed to combine financial datasets: {str(e)}")
        raise


def clean_and_transform_data(df):
    """
    Apply data cleaning and feature engineering transformations.
    
    This function performs comprehensive data cleaning including date parsing,
    missing value handling, and feature engineering for time-based analysis.
    
    Args:
        df (pd.DataFrame): Raw combined transaction data
        
    Returns:
        pd.DataFrame: Cleaned and transformed dataset ready for analysis
        
    Raises:
        ValueError: If critical data validation checks fail
    """
    logger.info("Starting data cleaning and transformation process")
    
    # Store initial record count for data loss tracking
    initial_record_count = len(df)
    
    # Parse and validate date fields
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Remove records with invalid dates
    df = df.dropna(subset=['Date'])
    records_after_date_cleaning = len(df)
    
    if records_after_date_cleaning < initial_record_count:
        dropped_records = initial_record_count - records_after_date_cleaning
        logger.warning(f"Removed {dropped_records} records with invalid dates")
    
    # Feature engineering for temporal analysis
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = df['Date'].dt.weekday >= 5
    df['Quarter'] = df['Date'].dt.quarter
    
    # Classify transactions by type based on amount direction
    df['Transaction_Type'] = df['Amount'].apply(
        lambda amount: 'Income' if amount > 0 else 'Expense'
    )
    
    # Sort by date to ensure chronological order for time series analysis
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Data validation checks
    date_range = df['Date'].max() - df['Date'].min()
    transaction_type_distribution = df['Transaction_Type'].value_counts()
    
    logger.info("Data cleaning and transformation completed")
    logger.info(f"Final dataset: {len(df):,} records")
    logger.info(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    logger.info(f"Analysis period: {date_range.days} days")
    logger.info(f"Transaction distribution: {transaction_type_distribution.to_dict()}")
    
    return df


def generate_data_quality_report(df):
    """
    Generate comprehensive data quality metrics and summary statistics.
    
    This function analyzes the cleaned dataset to provide insights into data completeness,
    quality, and key statistical measures for validation and documentation purposes.
    
    Args:
        df (pd.DataFrame): Cleaned transaction dataset
        
    Returns:
        dict: Comprehensive data quality metrics and summary statistics
    """
    logger.info("Generating data quality and summary report")
    
    # Separate transactions by type for targeted analysis
    expense_transactions = df[df['Transaction_Type'] == 'Expense']
    income_transactions = df[df['Transaction_Type'] == 'Income']
    categorized_transactions = df[df['Category'].notna()]
    
    # Calculate core business metrics
    summary_statistics = {
        'total_transactions': len(df),
        'analysis_period_days': (df['Date'].max() - df['Date'].min()).days,
        'unique_data_sources': df['Source'].nunique(),
        'expense_transaction_count': len(expense_transactions),
        'income_transaction_count': len(income_transactions),
        'categorized_transaction_count': len(categorized_transactions),
        'categorization_coverage_rate': len(categorized_transactions) / len(df) if len(df) > 0 else 0,
        'total_expense_amount': abs(expense_transactions['Amount'].sum()) if len(expense_transactions) > 0 else 0,
        'total_income_amount': income_transactions['Amount'].sum() if len(income_transactions) > 0 else 0,
        'unique_category_count': categorized_transactions['Category'].nunique() if len(categorized_transactions) > 0 else 0
    }
    
    # Calculate average monthly expenses for budgeting insights
    if len(expense_transactions) > 0:
        monthly_expenses = expense_transactions.groupby(
            expense_transactions['Date'].dt.to_period('M')
        )['Amount'].sum().abs()
        summary_statistics['average_monthly_expenses'] = monthly_expenses.mean()
    else:
        summary_statistics['average_monthly_expenses'] = 0
    
    # Data quality assessment metrics
    quality_metrics = {
        'missing_date_count': df['Date'].isnull().sum(),
        'missing_amount_count': df['Amount'].isnull().sum(),
        'missing_description_count': df['Description'].isnull().sum(),
        'missing_category_count': df['Category'].isnull().sum(),
        'duplicate_record_count': df.duplicated().sum()
    }
    
    # Combine all metrics
    comprehensive_report = {**summary_statistics, **quality_metrics}
    
    # Log key findings
    logger.info("Data quality report generated")
    logger.info(f"Data completeness: {len(df):,} total records processed")
    logger.info(f"Categorization rate: {comprehensive_report['categorization_coverage_rate']:.1%}")
    logger.info(f"Financial summary: ${comprehensive_report['total_expense_amount']:,.2f} expenses, "
               f"${comprehensive_report['total_income_amount']:,.2f} income")
    
    return comprehensive_report


def execute_data_pipeline():
    """
    Execute the complete data processing pipeline from raw files to analysis-ready dataset.
    
    This function orchestrates the entire data processing workflow including data loading,
    cleaning, transformation, and quality assessment. Handles errors gracefully and
    provides comprehensive logging for monitoring and debugging.
    
    Returns:
        tuple: (processed_dataframe, quality_report) or (None, None) if processing fails
    """
    logger.info("="*60)
    logger.info("PERSONAL FINANCE DATA PROCESSING PIPELINE EXECUTION")
    logger.info("="*60)
    
    try:
        # Step 1: Load and combine data from multiple sources
        combined_data = combine_financial_datasets()
        
        # Step 2: Clean and transform data for analysis
        processed_data = clean_and_transform_data(combined_data)
        
        # Step 3: Generate data quality assessment
        quality_report = generate_data_quality_report(processed_data)
        
        # Step 4: Persist processed data for downstream analysis
        output_filename = 'cleaned_financial_data.csv'
        processed_data.to_csv(output_filename, index=False)
        logger.info(f"Processed dataset saved to: {output_filename}")
        
        # Step 5: Save quality metrics for documentation
        quality_report_filename = 'data_quality_report.csv'
        pd.Series(quality_report).to_csv(quality_report_filename)
        logger.info(f"Data quality report saved to: {quality_report_filename}")
        
        logger.info("Data processing pipeline completed successfully")
        logger.info("Dataset is ready for analysis and machine learning")
        
        return processed_data, quality_report
        
    except FileNotFoundError:
        logger.error("Required data files not found")
        logger.error("Please ensure the following files exist in the working directory:")
        logger.error("  - wells_fargo_checking.csv")
        logger.error("  - discover_cc.csv") 
        logger.error("  - chase_cc.csv")
        return None, None
        
    except Exception as e:
        logger.error(f"Data processing pipeline failed: {str(e)}")
        logger.error("Please check input data format and file permissions")
        return None, None


if __name__ == "__main__":
    # Execute the data processing pipeline when script is run directly
    processed_dataset, data_quality_metrics = execute_data_pipeline()