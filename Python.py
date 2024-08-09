import pandas as pd
from datetime import datetime

# Define basic information
today = datetime(2013, 1, 22)  # Assume today's date is January 22, 2013
start_date = datetime(2013, 1, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)  # Fixed last published RPI date
last_rpi_value = 246.8  # Correct last published RPI value
annual_inflation_rate = 0.025  # Annual inflation rate
rpi_base = 135.1  # RPI base from October 1991
time_lag_months = 8  # Indexation lag of 8 months
unindexed_coupon = 2.0625  # Unindexed coupon payment per half-year

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 1:
        current_date = current_date.replace(month=7)
    else:
        current_date = current_date.replace(year=current_date.year + 1, month=1)

# Calculate indexation dates and related data
indexation_dates = []
time_intervals_years = []
projected_rpis = []
rpi_increase_factors = []
coupon_payments = []
time_to_payment_years = []
for date in dates:
    year = date.year if date.month > 8 else date.year - 1
    month = (date.month - 8) % 12 or 12
    indexation_date = datetime(year, month, 22)
    indexation_dates.append(indexation_date)

    # Calculate time interval in years
    days_interval = (indexation_date - last_published_rpi_date).days
    years_interval = max(days_interval / 365, 0)  # Use 365 days per year and ensure it's not less than 0
    time_intervals_years.append(years_interval)

    # Calculate projected RPI
    projected_rpi = last_rpi_value * (1 + annual_inflation_rate) ** years_interval
    projected_rpis.append(projected_rpi)

    # Calculate RPI increase factor
    rpi_increase_factor = projected_rpi / rpi_base
    rpi_increase_factors.append(rpi_increase_factor)

    # Calculate adjusted coupon payment
    coupon_payment = unindexed_coupon * rpi_increase_factor
    coupon_payments.append(coupon_payment)

    # Calculate time to payment date (in years)
    days_to_payment = (date - today).days
    years_to_payment = days_to_payment / 365
    time_to_payment_years.append(years_to_payment)

# Create DataFrame
payment_schedule = pd.DataFrame({
    'Payment Date': dates,
    'Indexation Date': indexation_dates,
    'Years to Indexation': time_intervals_years,
    'Projected RPI': projected_rpis,
    'RPI Increase Factor': rpi_increase_factors,
    'Coupon Money Payment': coupon_payments,
    'Time to Payment Date (years)': time_to_payment_years
})

# Display DataFrame
print(payment_schedule)

# Define basic information and data preparation
today = datetime(2013, 1, 22)
start_date = datetime(2013, 7, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)  # Fixed last published RPI date
last_rpi_value = 246.8  # Correct last published RPI value
annual_inflation_rate = 0.025  # Annual inflation rate
rpi_base = 135.1  # RPI base from October 1991
time_lag_months = 8  # Indexation lag of 8 months
unindexed_coupon = 2.0625  # Unindexed coupon payment per half-year
purchase_price = 323.57  # Bond purchase price
nominal_value = 100  # Nominal principal of the bond

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 7:
        current_date = current_date.replace(year=current_date.year + 1, month=1)
    else:
        current_date = current_date.replace(year=current_date.year, month=7)

# Calculate indexation dates and related data
indexation_dates = []
time_intervals_years = []
projected_rpis = []
rpi_increase_factors = []
coupon_payments = []
time_to_payment_years = []
for date in dates:
    year = date.year if date.month > 8 else date.year - 1
    month = (date.month - 8) % 12 or 12
    indexation_date = datetime(year, month, 22)
    indexation_dates.append(indexation_date)

    # Calculate time interval in years
    days_interval = (indexation_date - last_published_rpi_date).days
    years_interval = max(days_interval / 365, 0)  # Use 365 days per year, ensuring it is not less than 0
    time_intervals_years.append(years_interval)

    # Calculate projected RPI
    projected_rpi = last_rpi_value * (1 + annual_inflation_rate) ** years_interval
    projected_rpis.append(projected_rpi)

    # Calculate RPI increase factor
    rpi_increase_factor = projected_rpi / rpi_base
    rpi_increase_factors.append(rpi_increase_factor)

    # Calculate adjusted coupon payment
    coupon_payment = unindexed_coupon * rpi_increase_factor
    coupon_payments.append(coupon_payment)

    # Calculate time to payment date (in years)
    days_to_payment = (date - today).days
    years_to_payment = days_to_payment / 365
    time_to_payment_years.append(years_to_payment)

# Create DataFrame
payment_schedule = pd.DataFrame({
    'Payment Date': dates,
    'Indexation Date': indexation_dates,
    'Years to Indexation': time_intervals_years,
    'Projected RPI': projected_rpis,
    'RPI Increase Factor': rpi_increase_factors,
    'Coupon Money Payment': coupon_payments,
    'Time to Payment Date (years)': time_to_payment_years
})


# Display DataFrame
print("Payment Schedule DataFrame:")
print(payment_schedule)

# Extract future cash flows
cash_flows = [-purchase_price]  # Initial investment (purchase price) as negative cash flow
print("\nInitial Cash Flow (Purchase Price):")
print(cash_flows)

for payment_date, payment in zip(payment_schedule['Payment Date'], payment_schedule['Coupon Money Payment']):
    if payment_date > today:  # Exclude the coupon on January 22, 2013
        cash_flows.append(payment)

print("\nCash Flows After Adding Coupon Payments (Excluding 2013-01-22):")
print(cash_flows)

# Add the final principal redemption cash flow
final_redemption = nominal_value * payment_schedule['RPI Increase Factor'].iloc[-1]
cash_flows[-1] += final_redemption  # Add redemption principal to the last payment date
print("\nCash Flows After Adding Final Redemption:")
print(cash_flows)

# Calculate discount factor v
def npv(v):
    npv_value = 0
    for payment, t, payment_date in zip(payment_schedule['Coupon Money Payment'], payment_schedule['Time to Payment Date (years)'], payment_schedule['Payment Date']):
        if payment_date <= today:  # Exclude the coupon on January 22, 2013 and earlier
            continue
        npv_value += payment * v ** t
    npv_value += final_redemption * v ** payment_schedule['Time to Payment Date (years)'].iloc[-1]
    return npv_value - purchase_price

# Use root_scalar to find v
result = root_scalar(npv, bracket=[0, 1], method='brentq')
v = result.root
print(f"\nDiscount Factor (v): {v:.8f}")

# Calculate Money Yield
money_yield = (1 - v) / v
print(f"Money Yield: {money_yield:.8f}")

# Calculate Effective Annual Rate
effective_annual_rate = money_yield
print(f"\nMoney Yield (Effective Annual Rate of Interest): {effective_annual_rate:.8f}")


import numpy as np
import pandas as pd
from datetime import datetime

# Given data
money_yield = 0.02188507  # Result from Money yield calculation

# Assuming payment_schedule DataFrame is already defined and contains all necessary columns
# Update the DataFrame to include Discount Factors

# Calculate Discount Factor and update the DataFrame
payment_schedule['Discount Factor'] = (1 + money_yield) ** (-payment_schedule['Time to Payment Date (years)'])

# Display the updated DataFrame
print(payment_schedule[['Payment Date', 'Coupon Money Payment', 'Time to Payment Date (years)', 'Discount Factor']])


import numpy as np
import pandas as pd
from datetime import datetime

# Assume previous data and DataFrame are already prepared
# Assumed Money Yield value
money_yield = 0.02188507

# Update the DataFrame to include Discount Factors
payment_schedule['Discount Factor'] = (1 + money_yield) ** (-payment_schedule['Time to Payment Date (years)'])

# Calculate Discounted Coupon Money Payment
payment_schedule['Discounted Coupon Money Payment'] = payment_schedule['Coupon Money Payment'] * payment_schedule['Discount Factor']

# Display the updated DataFrame
print("Updated Payment Schedule with Discount Factors and Discounted Coupon Payments:")
print(payment_schedule[['Payment Date', 'Coupon Money Payment', 'Discount Factor', 'Discounted Coupon Money Payment']])


import numpy as np
import pandas as pd
from datetime import datetime

# Assuming previous data and DataFrame are already prepared
# Assumed Money Yield value
money_yield = 0.02188507

# Update the DataFrame to include Discount Factors
payment_schedule['Discount Factor'] = (1 + money_yield) ** (-payment_schedule['Time to Payment Date (years)'])

# Calculate Discounted Coupon Money Payment
payment_schedule['Discounted Coupon Money Payment'] = payment_schedule['Coupon Money Payment'] * payment_schedule['Discount Factor']

# Calculate the present value of the redemption payment
final_redemption_payment = nominal_value * payment_schedule['RPI Increase Factor'].iloc[-1]
discounted_redemption_payment = final_redemption_payment * payment_schedule['Discount Factor'].iloc[-1]

# Sum all discounted coupon payments
total_discounted_coupon_payments = payment_schedule['Discounted Coupon Money Payment'].sum()

# Total discounted money payment equals the sum of all discounted coupon payments and the discounted redemption payment
total_discounted_money_payment = total_discounted_coupon_payments + discounted_redemption_payment

print("Updated Payment Schedule with Discount Factors and Discounted Coupon Payments:")
print(payment_schedule[['Payment Date', 'Coupon Money Payment', 'Discounted Coupon Money Payment']])

print(f"\nTotal Discounted Coupon Payments: {total_discounted_coupon_payments:.2f}")
print(f"Discounted Redemption Money Payment: {discounted_redemption_payment:.2f}")
print(f"Total Discounted Money Payment: {total_discounted_money_payment:.2f}")


import numpy as np

# Given log-normal distribution parameters
mu = 0.0485
sigma = 0.0237

# First, sample from the normal distribution
# Q such that 1 + Q follows a log-normal distribution
# Thus, we sample from a normal distribution with mean mu and standard deviation sigma, then exponentiate the result

# Draw 20 values from the normal distribution
normal_samples = np.random.normal(mu, sigma, 20)

# Convert to log-normal samples for 1 + Q
log_normal_samples = np.exp(normal_samples)

# Since 1 + Q is log-normal, subtract 1 to obtain Q
q_samples = log_normal_samples - 1

# Output the 20 sampled Q values
print(q_samples)

import numpy as np

# Given log-normal distribution parameters
mu = 0.0485
sigma = 0.0237

# First, sample from the normal distribution
# Q such that 1 + Q follows a log-normal distribution
# Thus, we sample from a normal distribution with mean mu and standard deviation sigma, then exponentiate the result

# Draw 50 values from the normal distribution
normal_samples = np.random.normal(mu, sigma, 50)

# Convert to log-normal samples for 1 + Q
log_normal_samples = np.exp(normal_samples)

# Since 1 + Q is log-normal, subtract 1 to obtain Q
q_samples = log_normal_samples - 1

# Output the 50 sampled Q values
print(q_samples)


import numpy as np
import pandas as pd
from datetime import datetime

# For demonstration, assume the following initial settings
nominal_value = 100  # Assume the bond nominal value is 100 GBP
start_date = datetime(2013, 1, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)
last_rpi_value = 246.8
rpi_base = 135.1
time_lag_months = 8
unindexed_coupon = 2.0625
money_yield = 0.02188507

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 1:
        current_date = current_date.replace(month=7)
    else:
        current_date = current_date.replace(year=current_date.year + 1, month=1)

# Use the previously generated 20 random Q values
q_values = [0.0642, 0.0521, 0.0535, 0.0725, 0.0196, 0.0516, 0.0414, 0.0247, 0.0544, 0.0391, 0.0493, 0.0651, 0.0414, 0.0239, 0.0960, 0.0748, 0.0511, 0.0825, 0.0657, 0.0553]
 
results = []
for q in q_values:
    projected_rpis = []
    rpi_increase_factors = []
    coupon_payments = []
    discount_factors = []
    discounted_coupon_payments = []
    
    for date in dates:
        indexation_date = datetime(date.year if date.month > 8 else date.year - 1, (date.month - 8) % 12 or 12, 22)
        years_interval = (indexation_date - last_published_rpi_date).days / 365
        projected_rpi = last_rpi_value * (1 + q) ** years_interval
        projected_rpis.append(projected_rpi)
        rpi_increase_factor = projected_rpi / rpi_base
        rpi_increase_factors.append(rpi_increase_factor)
        coupon_payment = unindexed_coupon * rpi_increase_factor
        coupon_payments.append(coupon_payment)
        years_to_payment = (date - start_date).days / 365
        discount_factor = (1 + money_yield) ** (-years_to_payment)
        discount_factors.append(discount_factor)
        discounted_coupon_payment = coupon_payment * discount_factor
        discounted_coupon_payments.append(discounted_coupon_payment)
    
    final_redemption_payment = nominal_value * rpi_increase_factors[-1]
    discounted_redemption_payment = final_redemption_payment * discount_factors[-1]
    total_discounted_coupon_payments = sum(discounted_coupon_payments)
    total_discounted_money_payment = total_discounted_coupon_payments + discounted_redemption_payment
    
    results.append({
        'Q': q,
        'Total Discounted Coupon Payments': total_discounted_coupon_payments,
        'Discounted Redemption Payment': discounted_redemption_payment,
        'Total Discounted Money Payment': total_discounted_money_payment
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)

# Calculate the mean and standard deviation
mean_total_discounted_payments = results_df['Total Discounted Money Payment'].mean()
std_total_discounted_payments = results_df['Total Discounted Money Payment'].std()

# Output the results
print(mean_total_discounted_payments)
print(std_total_discounted_payments)

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Initial settings
nominal_value = 100  # Bond nominal value
start_date = datetime(2013, 1, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)
last_rpi_value = 246.8
rpi_base = 135.1
time_lag_months = 8
unindexed_coupon = 2.0625
money_yield = 0.02188507

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 1:
        current_date = current_date.replace(month=7)
    else:
        current_date = current_date.replace(year=current_date.year + 1, month=1)

# Use the previously generated 20 random Q values
q_values = [0.0642, 0.0521, 0.0535, 0.0725, 0.0196, 0.0516, 0.0414, 0.0247, 0.0544, 0.0391, 0.0493, 0.0651, 0.0414, 0.0239, 0.0960, 0.0748, 0.0511, 0.0825, 0.0657, 0.0553]

results = []
for q in q_values:
    projected_rpis = []
    rpi_increase_factors = []
    coupon_payments = []
    discount_factors = []
    discounted_coupon_payments = []
    
    for date in dates:
        indexation_date = datetime(date.year if date.month > 8 else date.year - 1, (date.month - 8) % 12 or 12, 22)
        years_interval = (indexation_date - last_published_rpi_date).days / 365
        projected_rpi = last_rpi_value * (1 + q) ** years_interval
        projected_rpis.append(projected_rpi)
        rpi_increase_factor = projected_rpi / rpi_base
        rpi_increase_factors.append(rpi_increase_factor)
        coupon_payment = unindexed_coupon * rpi_increase_factor
        coupon_payments.append(coupon_payment)
        years_to_payment = (date - start_date).days / 365
        discount_factor = (1 + money_yield) ** (-years_to_payment)
        discount_factors.append(discount_factor)
        discounted_coupon_payment = coupon_payment * discount_factor
        discounted_coupon_payments.append(discounted_coupon_payment)
    
    final_redemption_payment = nominal_value * rpi_increase_factors[-1]
    discounted_redemption_payment = final_redemption_payment * discount_factors[-1]
    total_discounted_coupon_payments = sum(discounted_coupon_payments)
    total_discounted_money_payment = total_discounted_coupon_payments + discounted_redemption_payment
    
    results.append({
        'Q': q,
        'Total Discounted Coupon Payments': total_discounted_coupon_payments,
        'Discounted Redemption Payment': discounted_redemption_payment,
        'Total Discounted Money Payment': total_discounted_money_payment
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Q'], results_df['Total Discounted Money Payment'], color='blue', label='Total Discounted Money Payment')

# Add title and labels
plt.title('Total Discounted Money Payment vs Inflation Rate (Q)')
plt.xlabel('Inflation Rate (Q)')
plt.ylabel('Total Discounted Money Payment')

# Show legend
plt.legend()

# Save the figure
plt.savefig('/Users/shuojiehu/Desktop/Total_Discounted_Money_Payment_vs_Inflation_Rate.png')

# Show plot
plt.show()

import numpy as np

# Given log-normal distribution parameters
mu = 0.015
sigma = 0.005

# First, sample from the normal distribution
# Q such that 1 + Q follows a log-normal distribution
# Thus, we sample from a normal distribution with mean mu and standard deviation sigma, then exponentiate the result

# Draw 50 values from the normal distribution
normal_samples = np.random.normal(mu, sigma, 50)

# Convert to log-normal samples for 1 + Q
log_normal_samples = np.exp(normal_samples)

# Since 1 + Q is log-normal, subtract 1 to obtain Q
q_samples = log_normal_samples - 1

# Output the 50 sampled Q values
print(q_samples)

import numpy as np
import pandas as pd
from datetime import datetime

# For demonstration, assume the following initial settings
nominal_value = 100  # Assume the bond nominal value is 100 GBP
start_date = datetime(2013, 1, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)
last_rpi_value = 246.8
rpi_base = 135.1
time_lag_months = 8
unindexed_coupon = 2.0625
money_yield = 0.02188507

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 1:
        current_date = current_date.replace(month=7)
    else:
        current_date = current_date.replace(year=current_date.year + 1, month=1)

# Use the previously generated 50 random Q values
q_values = [
    0.01174024, 0.00476118, 0.01242266, 0.00869116, 0.01310452, 0.01299418,
    0.01995747, 0.02220736, 0.01951094, 0.0149184,  0.00623809, 0.00875912,
    0.01435892, 0.01055972, 0.01579924, 0.0132766,  0.01666278, 0.01606621,
    0.02291391, 0.01573701, 0.0216858,  0.01383781, 0.02743413, 0.01248873,
    0.01381837, 0.01702032, 0.0159719,  0.01811564, 0.01927372, 0.01912719,
    0.01943147, 0.02169679, 0.01794286, 0.01235298, 0.01586704, 0.01364475,
    0.0211385,  0.01022694, 0.01341142, 0.02057492, 0.01870596, 0.01534625,
    0.01649501, 0.01530854, 0.021296,   0.01072791, 0.01240227, 0.01173456,
    0.01965593, 0.00961522
]

results = []
for q in q_values:
    projected_rpis = []
    rpi_increase_factors = []
    coupon_payments = []
    discount_factors = []
    discounted_coupon_payments = []
    
    for date in dates:
        indexation_date = datetime(date.year if date.month > 8 else date.year - 1, (date.month - 8) % 12 or 12, 22)
        years_interval = (indexation_date - last_published_rpi_date).days / 365
        projected_rpi = last_rpi_value * (1 + q) ** years_interval
        projected_rpis.append(projected_rpi)
        rpi_increase_factor = projected_rpi / rpi_base
        rpi_increase_factors.append(rpi_increase_factor)
        coupon_payment = unindexed_coupon * rpi_increase_factor
        coupon_payments.append(coupon_payment)
        years_to_payment = (date - start_date).days / 365
        discount_factor = (1 + money_yield) ** (-years_to_payment)
        discount_factors.append(discount_factor)
        discounted_coupon_payment = coupon_payment * discount_factor
        discounted_coupon_payments.append(discounted_coupon_payment)
    
    final_redemption_payment = nominal_value * rpi_increase_factors[-1]
    discounted_redemption_payment = final_redemption_payment * discount_factors[-1]
    total_discounted_coupon_payments = sum(discounted_coupon_payments)
    total_discounted_money_payment = total_discounted_coupon_payments + discounted_redemption_payment
    
    results.append({
        'Q': q,
        'Total Discounted Coupon Payments': total_discounted_coupon_payments,
        'Discounted Redemption Payment': discounted_redemption_payment,
        'Total Discounted Money Payment': total_discounted_money_payment
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)


# Calculate the mean and standard deviation
mean_total_discounted_payments = results_df['Total Discounted Money Payment'].mean()
std_total_discounted_payments = results_df['Total Discounted Money Payment'].std()

# Output the results
print(mean_total_discounted_payments)
print(std_total_discounted_payments)

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Initial settings
nominal_value = 100  # Bond nominal value
start_date = datetime(2013, 1, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)
last_rpi_value = 246.8
rpi_base = 135.1
time_lag_months = 8
unindexed_coupon = 2.0625
money_yield = 0.02188507

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 1:
        current_date = current_date.replace(month=7)
    else:
        current_date = current_date.replace(year=current_date.year + 1, month=1)

# Use the previously generated 50 random Q values
q_values = [
    0.01174024, 0.00476118, 0.01242266, 0.00869116, 0.01310452, 0.01299418,
    0.01995747, 0.02220736, 0.01951094, 0.0149184,  0.00623809, 0.00875912,
    0.01435892, 0.01055972, 0.01579924, 0.0132766,  0.01666278, 0.01606621,
    0.02291391, 0.01573701, 0.0216858,  0.01383781, 0.02743413, 0.01248873,
    0.01381837, 0.01702032, 0.0159719,  0.01811564, 0.01927372, 0.01912719,
    0.01943147, 0.02169679, 0.01794286, 0.01235298, 0.01586704, 0.01364475,
    0.0211385,  0.01022694, 0.01341142, 0.02057492, 0.01870596, 0.01534625,
    0.01649501, 0.01530854, 0.021296,   0.01072791, 0.01240227, 0.01173456,
    0.01965593, 0.00961522
]

results = []
for q in q_values:
    projected_rpis = []
    rpi_increase_factors = []
    coupon_payments = []
    discount_factors = []
    discounted_coupon_payments = []
    
    for date in dates:
        indexation_date = datetime(date.year if date.month > 8 else date.year - 1, (date.month - 8) % 12 or 12, 22)
        years_interval = (indexation_date - last_published_rpi_date).days / 365
        projected_rpi = last_rpi_value * (1 + q) ** years_interval
        projected_rpis.append(projected_rpi)
        rpi_increase_factor = projected_rpi / rpi_base
        rpi_increase_factors.append(rpi_increase_factor)
        coupon_payment = unindexed_coupon * rpi_increase_factor
        coupon_payments.append(coupon_payment)
        years_to_payment = (date - start_date).days / 365
        discount_factor = (1 + money_yield) ** (-years_to_payment)
        discount_factors.append(discount_factor)
        discounted_coupon_payment = coupon_payment * discount_factor
        discounted_coupon_payments.append(discounted_coupon_payment)
    
    final_redemption_payment = nominal_value * rpi_increase_factors[-1]
    discounted_redemption_payment = final_redemption_payment * discount_factors[-1]
    total_discounted_coupon_payments = sum(discounted_coupon_payments)
    total_discounted_money_payment = total_discounted_coupon_payments + discounted_redemption_payment
    
    results.append({
        'Q': q,
        'Total Discounted Coupon Payments': total_discounted_coupon_payments,
        'Discounted Redemption Payment': discounted_redemption_payment,
        'Total Discounted Money Payment': total_discounted_money_payment
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Q'], results_df['Total Discounted Money Payment'], color='blue', label='Total Discounted Money Payment')

# Add title and labels
plt.title('Total Discounted Money Payment vs Inflation Rate (Q)')
plt.xlabel('Inflation Rate (Q)')
plt.ylabel('Total Discounted Money Payment')

# Show legend
plt.legend()

# Save the figure
plt.savefig('/Users/shuojiehu/Desktop/Total_Discounted_Money_Payment_vs_Inflation_Rate.png')

# Show plot
plt.show()


import numpy as np
import pandas as pd
from datetime import datetime

# For demonstration, assume the following initial settings
nominal_value = 100  # Assume the bond nominal value is 100 GBP
start_date = datetime(2013, 1, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)
last_rpi_value = 246.8
rpi_base = 135.1
time_lag_months = 8
unindexed_coupon = 2.0625
money_yield = 0.02188507

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 1:
        current_date = current_date.replace(month=7)
    else:
        current_date = current_date.replace(year=current_date.year + 1, month=1)

# Use the previously generated 50 random Q values
q_values = [
    0.03696943, 0.03662748, 0.03695559, 0.02813381, 0.08390984, 0.05002338,
    0.04000184, 0.03394804, 0.06219634, 0.0657016,  0.06420962, 0.04451735,
    0.06125127, 0.07038442, 0.02850313, 0.08435148, 0.07023999, 0.04908765,
    0.08416722, 0.06661712, 0.00660709, 0.02779188, 0.01513378, 0.04389103,
    0.03272674, 0.09472572, 0.00517663, 0.06773728, 0.02585263, 0.02806562,
    0.05610093, 0.0859817,  0.04611161, 0.0605972,  0.0446469,  0.06776839,
    0.05275233, 0.08136871, 0.00550237, 0.06323708, 0.01951862, 0.01519869,
    0.01959568, 0.068936,   0.02659081, 0.06352124, 0.05781016, 0.03835776,
    0.0668442,  0.08091752
]
 
results = []
for q in q_values:
    projected_rpis = []
    rpi_increase_factors = []
    coupon_payments = []
    discount_factors = []
    discounted_coupon_payments = []
    
    for date in dates:
        indexation_date = datetime(date.year if date.month > 8 else date.year - 1, (date.month - 8) % 12 or 12, 22)
        years_interval = (indexation_date - last_published_rpi_date).days / 365
        projected_rpi = last_rpi_value * (1 + q) ** years_interval
        projected_rpis.append(projected_rpi)
        rpi_increase_factor = projected_rpi / rpi_base
        rpi_increase_factors.append(rpi_increase_factor)
        coupon_payment = unindexed_coupon * rpi_increase_factor
        coupon_payments.append(coupon_payment)
        years_to_payment = (date - start_date).days / 365
        discount_factor = (1 + money_yield) ** (-years_to_payment)
        discount_factors.append(discount_factor)
        discounted_coupon_payment = coupon_payment * discount_factor
        discounted_coupon_payments.append(discounted_coupon_payment)
    
    final_redemption_payment = nominal_value * rpi_increase_factors[-1]
    discounted_redemption_payment = final_redemption_payment * discount_factors[-1]
    total_discounted_coupon_payments = sum(discounted_coupon_payments)
    total_discounted_money_payment = total_discounted_coupon_payments + discounted_redemption_payment
    
    results.append({
        'Q': q,
        'Total Discounted Coupon Payments': total_discounted_coupon_payments,
        'Discounted Redemption Payment': discounted_redemption_payment,
        'Total Discounted Money Payment': total_discounted_money_payment
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)

# Calculate the mean and standard deviation
mean_total_discounted_payments = results_df['Total Discounted Money Payment'].mean()
std_total_discounted_payments = results_df['Total Discounted Money Payment'].std()

# Output the results
print(mean_total_discounted_payments)
print(std_total_discounted_payments)

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Initial settings
nominal_value = 100  # Bond nominal value
start_date = datetime(2013, 1, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)
last_rpi_value = 246.8
rpi_base = 135.1
time_lag_months = 8
unindexed_coupon = 2.0625
money_yield = 0.02188507

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 1:
        current_date = current_date.replace(month=7)
    else:
        current_date = current_date.replace(year=current_date.year + 1, month=1)

# Use the previously generated 50 random Q values
q_values = [
    0.03696943, 0.03662748, 0.03695559, 0.02813381, 0.08390984, 0.05002338,
    0.04000184, 0.03394804, 0.06219634, 0.0657016,  0.06420962, 0.04451735,
    0.06125127, 0.07038442, 0.02850313, 0.08435148, 0.07023999, 0.04908765,
    0.08416722, 0.06661712, 0.00660709, 0.02779188, 0.01513378, 0.04389103,
    0.03272674, 0.09472572, 0.00517663, 0.06773728, 0.02585263, 0.02806562,
    0.05610093, 0.0859817,  0.04611161, 0.0605972,  0.0446469,  0.06776839,
    0.05275233, 0.08136871, 0.00550237, 0.06323708, 0.01951862, 0.01519869,
    0.01959568, 0.068936,   0.02659081, 0.06352124, 0.05781016, 0.03835776,
    0.0668442,  0.08091752
]

results = []
for q in q_values:
    projected_rpis = []
    rpi_increase_factors = []
    coupon_payments = []
    discount_factors = []
    discounted_coupon_payments = []
    
    for date in dates:
        indexation_date = datetime(date.year if date.month > 8 else date.year - 1, (date.month - 8) % 12 or 12, 22)
        years_interval = (indexation_date - last_published_rpi_date).days / 365
        projected_rpi = last_rpi_value * (1 + q) ** years_interval
        projected_rpis.append(projected_rpi)
        rpi_increase_factor = projected_rpi / rpi_base
        rpi_increase_factors.append(rpi_increase_factor)
        coupon_payment = unindexed_coupon * rpi_increase_factor
        coupon_payments.append(coupon_payment)
        years_to_payment = (date - start_date).days / 365
        discount_factor = (1 + money_yield) ** (-years_to_payment)
        discount_factors.append(discount_factor)
        discounted_coupon_payment = coupon_payment * discount_factor
        discounted_coupon_payments.append(discounted_coupon_payment)
    
    final_redemption_payment = nominal_value * rpi_increase_factors[-1]
    discounted_redemption_payment = final_redemption_payment * discount_factors[-1]
    total_discounted_coupon_payments = sum(discounted_coupon_payments)
    total_discounted_money_payment = total_discounted_coupon_payments + discounted_redemption_payment
    
    results.append({
        'Q': q,
        'Total Discounted Coupon Payments': total_discounted_coupon_payments,
        'Discounted Redemption Payment': discounted_redemption_payment,
        'Total Discounted Money Payment': total_discounted_money_payment
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Q'], results_df['Total Discounted Money Payment'], color='blue', label='Total Discounted Money Payment')

# Add title and labels
plt.title('Total Discounted Money Payment vs Inflation Rate (Q)')
plt.xlabel('Inflation Rate (Q)')
plt.ylabel('Total Discounted Money Payment')

# Show legend
plt.legend()

# Save the figure
plt.savefig('/Users/shuojiehu/Desktop/Total_Discounted_Money_Payment_vs_Inflation_Rate2.png')

# Show plot
plt.show()

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import root_scalar

# Define basic information and data preparation
today = datetime(2013, 1, 22)
start_date = datetime(2013, 7, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)  # Fixed last published RPI date
last_rpi_value = 246.8  # Correct last published RPI value
rpi_base = 135.1  # RPI base from October 1991
time_lag_months = 8  # Indexation lag of 8 months
unindexed_coupon = 2.0625  # Unindexed coupon amount per half year
purchase_price = 323.57  # Bond purchase price
nominal_value = 100  # Nominal principal of the bond

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 7:
        current_date = current_date.replace(year=current_date.year + 1, month=1)
    else:
        current_date = current_date.replace(year=current_date.year, month=7)

# Function to calculate discount factor and yield
def calculate_discount_factor_and_yield(annual_inflation_rate):
    indexation_dates = []
    time_intervals_years = []
    projected_rpis = []
    rpi_increase_factors = []
    coupon_payments = []
    time_to_payment_years = []
    
    for date in dates:
        year = date.year if date.month > 8 else date.year - 1
        month = (date.month - 8) % 12 or 12
        indexation_date = datetime(year, month, 22)
        indexation_dates.append(indexation_date)

        # Calculate time interval in years
        days_interval = (indexation_date - last_published_rpi_date).days
        years_interval = max(days_interval / 365, 0)  # Use 365 days per year and ensure non-negative
        time_intervals_years.append(years_interval)

        # Calculate projected RPI
        projected_rpi = last_rpi_value * (1 + annual_inflation_rate) ** years_interval
        projected_rpis.append(projected_rpi)

        # Calculate RPI increase factor
        rpi_increase_factor = projected_rpi / rpi_base
        rpi_increase_factors.append(rpi_increase_factor)

        # Calculate adjusted coupon payment
        coupon_payment = unindexed_coupon * rpi_increase_factor
        coupon_payments.append(coupon_payment)

        # Calculate time to payment date in years
        days_to_payment = (date - today).days
        years_to_payment = days_to_payment / 365
        time_to_payment_years.append(years_to_payment)

    # Create DataFrame
    payment_schedule = pd.DataFrame({
        'Payment Date': dates,
        'Indexation Date': indexation_dates,
        'Years to Indexation': time_intervals_years,
        'Projected RPI': projected_rpis,
        'RPI Increase Factor': rpi_increase_factors,
        'Coupon Money Payment': coupon_payments,
        'Time to Payment Date (years)': time_to_payment_years
    })

    # Extract future cash flows
    cash_flows = [-purchase_price]  # Initial investment (purchase price) as negative cash flow
    for payment_date, payment in zip(payment_schedule['Payment Date'], payment_schedule['Coupon Money Payment']):
        if payment_date > today:  # Exclude coupon on 22 January 2013
            cash_flows.append(payment)

    # Add final redemption cash flow
    final_redemption = nominal_value * payment_schedule['RPI Increase Factor'].iloc[-1]
    cash_flows[-1] += final_redemption  # Add redemption to last payment date

    # Calculate discount factor v
    def npv(v):
        npv_value = 0
        for payment, t, payment_date in zip(payment_schedule['Coupon Money Payment'], payment_schedule['Time to Payment Date (years)'], payment_schedule['Payment Date']):
            if payment_date <= today:  # Exclude coupons on or before 22 January 2013
                continue
            npv_value += payment * v ** t
        npv_value += final_redemption * v ** payment_schedule['Time to Payment Date (years)'].iloc[-1]
        return npv_value - purchase_price

    # Use root_scalar to find v
    result = root_scalar(npv, bracket=[0, 1], method='brentq')
    v = result.root

    # Calculate Money Yield
    money_yield = (1 - v) / v

    return money_yield

# Calculate Money Yield for different inflation rates
q_values = [0.0642, 0.0521, 0.0535, 0.0725, 0.0196, 0.0516, 0.0414, 0.0247, 0.0544, 0.0391, 0.0493, 0.0651, 0.0414, 0.0239, 0.0960, 0.0748, 0.0511, 0.0825, 0.0657, 0.0553]
money_yields = [calculate_discount_factor_and_yield(q) for q in q_values]

# Display different inflation rates and Money Yield
inflation_yield_df = pd.DataFrame({
    'Annual Inflation Rate': q_values,
    'Money Yield': money_yields
})

print(inflation_yield_df)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Define basic information and data preparation
today = datetime(2013, 1, 22)
start_date = datetime(2013, 7, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)  # Fixed last published RPI date
last_rpi_value = 246.8  # Correct last published RPI value
rpi_base = 135.1  # RPI base from October 1991
time_lag_months = 8  # Indexation lag of 8 months
unindexed_coupon = 2.0625  # Unindexed coupon amount per half year
purchase_price = 323.57  # Bond purchase price
nominal_value = 100  # Nominal principal of the bond

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 7:
        current_date = current_date.replace(year=current_date.year + 1, month=1)
    else:
        current_date = current_date.replace(year=current_date.year, month=7)

# Function to calculate discount factor and yield
def calculate_discount_factor_and_yield(annual_inflation_rate):
    indexation_dates = []
    time_intervals_years = []
    projected_rpis = []
    rpi_increase_factors = []
    coupon_payments = []
    time_to_payment_years = []
    
    for date in dates:
        year = date.year if date.month > 8 else date.year - 1
        month = (date.month - 8) % 12 or 12
        indexation_date = datetime(year, month, 22)
        indexation_dates.append(indexation_date)

        # Calculate time interval in years
        days_interval = (indexation_date - last_published_rpi_date).days
        years_interval = max(days_interval / 365, 0)  # Use 365 days per year and ensure non-negative
        time_intervals_years.append(years_interval)

        # Calculate projected RPI
        projected_rpi = last_rpi_value * (1 + annual_inflation_rate) ** years_interval
        projected_rpis.append(projected_rpi)

        # Calculate RPI increase factor
        rpi_increase_factor = projected_rpi / rpi_base
        rpi_increase_factors.append(rpi_increase_factor)

        # Calculate adjusted coupon payment
        coupon_payment = unindexed_coupon * rpi_increase_factor
        coupon_payments.append(coupon_payment)

        # Calculate time to payment date in years
        days_to_payment = (date - today).days
        years_to_payment = days_to_payment / 365
        time_to_payment_years.append(years_to_payment)

    # Create DataFrame
    payment_schedule = pd.DataFrame({
        'Payment Date': dates,
        'Indexation Date': indexation_dates,
        'Years to Indexation': time_intervals_years,
        'Projected RPI': projected_rpis,
        'RPI Increase Factor': rpi_increase_factors,
        'Coupon Money Payment': coupon_payments,
        'Time to Payment Date (years)': time_to_payment_years
    })

    # Extract future cash flows
    cash_flows = [-purchase_price]  # Initial investment (purchase price) as negative cash flow
    for payment_date, payment in zip(payment_schedule['Payment Date'], payment_schedule['Coupon Money Payment']):
        if payment_date > today:  # Exclude coupon on 22 January 2013
            cash_flows.append(payment)

    # Add final redemption cash flow
    final_redemption = nominal_value * payment_schedule['RPI Increase Factor'].iloc[-1]
    cash_flows[-1] += final_redemption  # Add redemption to last payment date

    # Calculate discount factor v
    def npv(v):
        npv_value = 0
        for payment, t, payment_date in zip(payment_schedule['Coupon Money Payment'], payment_schedule['Time to Payment Date (years)'], payment_schedule['Payment Date']):
            if payment_date <= today:  # Exclude coupons on or before 22 January 2013
                continue
            npv_value += payment * v ** t
        npv_value += final_redemption * v ** payment_schedule['Time to Payment Date (years)'].iloc[-1]
        return npv_value - purchase_price

    # Use root_scalar to find v
    result = root_scalar(npv, bracket=[0, 1], method='brentq')
    v = result.root

    # Calculate Money Yield
    money_yield = (1 - v) / v

    return money_yield

# Calculate Money Yield for different inflation rates
q_values = [0.0642, 0.0521, 0.0535, 0.0725, 0.0196, 0.0516, 0.0414, 0.0247, 0.0544, 0.0391, 0.0493, 0.0651, 0.0414, 0.0239, 0.0960, 0.0748, 0.0511, 0.0825, 0.0657, 0.0553]
money_yields = [calculate_discount_factor_and_yield(q) for q in q_values]

# Display different inflation rates and Money Yield
inflation_yield_df = pd.DataFrame({
    'Annual Inflation Rate': q_values,
    'Money Yield': money_yields
})

# Polynomial fit
poly = Polynomial.fit(inflation_yield_df['Annual Inflation Rate'], inflation_yield_df['Money Yield'], 2)

# Generate points for the fitted line
x_fit = np.linspace(min(inflation_yield_df['Annual Inflation Rate']), max(inflation_yield_df['Annual Inflation Rate']), 100)
y_fit = poly(x_fit)

# Create a scatter plot of Money Yield vs Inflation Rate
plt.figure(figsize=(10, 6))
plt.scatter(inflation_yield_df['Annual Inflation Rate'], inflation_yield_df['Money Yield'], color='blue', label='Data')
plt.plot(x_fit, y_fit, color='red', label='Polynomial fit')
plt.title('Money Yield vs Annual Inflation Rate')
plt.xlabel('Annual Inflation Rate')
plt.ylabel('Money Yield')
plt.legend()
plt.grid(True)
plt.show()

# Display the polynomial coefficients
poly.convert().coef


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import root_scalar

# Redefine necessary imports and variables for the calculations
today = datetime(2013, 1, 22)
start_date = datetime(2013, 7, 22)
end_date = datetime(2030, 7, 22)
last_published_rpi_date = datetime(2012, 12, 22)
last_rpi_value = 246.8
rpi_base = 135.1
time_lag_months = 8
unindexed_coupon = 2.0625
purchase_price = 323.57
nominal_value = 100

# Generate payment dates
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    if current_date.month == 7:
        current_date = current_date.replace(year=current_date.year + 1, month=1)
    else:
        current_date = current_date.replace(year=current_date.year, month=7)

# Calculate time_to_payment_years
time_to_payment_years = []
for date in dates:
    days_to_payment = (date - today).days
    years_to_payment = days_to_payment / 365
    time_to_payment_years.append(years_to_payment)

# Define the function to calculate money yield
def calculate_money_yield(inflation_rate):
    projected_rpis = []
    rpi_increase_factors = []
    coupon_payments = []
    real_payments = []
    ratio_rpis = []
    
    for date in dates:
        year = date.year if date.month > 8 else date.year - 1
        month = (date.month - 8) % 12 or 12
        indexation_date = datetime(year, month, 22)

        days_interval = (indexation_date - last_published_rpi_date).days
        years_interval = max(days_interval / 365, 0)
        
        projected_rpi = last_rpi_value * (1 + inflation_rate) ** years_interval
        projected_rpis.append(projected_rpi)
        
        rpi_increase_factor = projected_rpi / rpi_base
        rpi_increase_factors.append(rpi_increase_factor)
        
        coupon_payment = unindexed_coupon * rpi_increase_factor
        coupon_payments.append(coupon_payment)
        
        days_to_payment = (date - today).days
        years_to_payment = days_to_payment / 365
        
        ratio_rpi = (1 + inflation_rate) ** -years_to_payment
        ratio_rpis.append(ratio_rpi)
        
        real_payment = coupon_payment * ratio_rpi
        real_payments.append(real_payment)

    cash_flows = [-purchase_price]
    for payment_date, payment in zip(dates, real_payments):
        if payment_date > today:
            cash_flows.append(payment)

    final_redemption = nominal_value * rpi_increase_factors[-1] * ratio_rpis[-1]
    cash_flows[-1] += final_redemption

    def npv(v):
        npv_value = 0
        for payment, t, payment_date in zip(real_payments, time_to_payment_years, dates):
            if payment_date <= today:
                continue
            npv_value += payment * v ** t
        npv_value += final_redemption * v ** time_to_payment_years[-1]
        return npv_value - purchase_price

    result = root_scalar(npv, bracket=[0, 2], method='brentq')
    v = result.root
    money_yield = (1 - v) / v
    return money_yield

# Define the inflation rates
inflation_rates = [0.0642, 0.0521, 0.0535, 0.0725, 0.0196, 0.0516, 0.0414, 0.0247, 0.0544, 0.0391, 
                   0.0493, 0.0651, 0.0414, 0.0239, 0.0960, 0.0748, 0.0511, 0.0825, 0.0657, 0.0553]

# Calculate money yield for each inflation rate
money_yields = []
for rate in inflation_rates:
    yield_value = calculate_money_yield(rate)
    money_yields.append(yield_value)

# Output results
money_yields_df = pd.DataFrame({
    'Inflation Rate': inflation_rates,
    'REAL Yield': money_yields
})

# Display the DataFrame
money_yields_df










