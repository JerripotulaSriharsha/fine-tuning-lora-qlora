def format_credit_risk_input(age, occupation, annual_income, credit_utilization, 
                            outstanding_debt, payment_behavior, credit_mix):
    """
    Convert raw credit risk features into formatted input for LLM fine-tuning
    
    Args:
        age: Customer age (int)
        occupation: Customer occupation (str)
        annual_income: Annual income amount (float)
        credit_utilization: Credit utilization ratio 0-100 (float)
        outstanding_debt: Outstanding debt amount (float)
        payment_behavior: Payment behavior pattern (str)
        credit_mix: Credit mix type (str)
    
    Returns:
        Formatted input string for LLM fine-tuning
    """
    
    # Calculate debt-to-income ratio
    dti_ratio = (outstanding_debt / annual_income) * 100 if annual_income > 0 else 0
    
    # Format currency values
    income_formatted = f"${annual_income:,.2f}"
    debt_formatted = f"${outstanding_debt:,.2f}"
    
    # Format percentage values
    util_formatted = f"{credit_utilization:.1f}%"
    dti_formatted = f"{dti_ratio:.1f}%"
    
    # Create the formatted input as a simple string
    formatted_input = f"Age: {age}, Occupation: {occupation}, Annual Income: {annual_income}, Outstanding Debt: {outstanding_debt}, Credit Utilization Ratio: {credit_utilization}, Payment Behaviour: {payment_behavior}"
    
    return formatted_input


def process_dataset_row(row):
    """
    Process a single row from your dataset.csv
    
    Args:
        row: Dictionary or pandas Series containing row data
    
    Returns:
        Formatted input string
    """
    return format_credit_risk_input(
        age=row['Age'],
        occupation=row['Occupation'],
        annual_income=float(row['Annual_Income']),
        credit_utilization=float(row['Credit_Utilization_Ratio']),
        outstanding_debt=float(row['Outstanding_Debt']),
        payment_behavior=row['Payment_Behaviour'],
        credit_mix=row['Credit_Mix']
    )


# Example usage
if __name__ == "__main__":
    # Your exact example
    example_data = {
        'Age': 32,
        'Occupation': 'Journalist',
        'Annual_Income': 33470.43,
        'Credit_Utilization_Ratio': 26.8,
        'Outstanding_Debt': 1318.49,
        'Payment_Behaviour': 'High_spent_Small_value_payments',
        'Credit_Mix': 'Standard'
    }
    
    # Generate formatted input
    result = format_credit_risk_input(
        age=example_data['Age'],
        occupation=example_data['Occupation'],
        annual_income=example_data['Annual_Income'],
        credit_utilization=example_data['Credit_Utilization_Ratio'],
        outstanding_debt=example_data['Outstanding_Debt'],
        payment_behavior=example_data['Payment_Behaviour'],
        credit_mix=example_data['Credit_Mix']
    )
    
    print("GENERATED FORMATTED INPUT:")
    print("=" * 80)
    print(result)
    print("=" * 80)
