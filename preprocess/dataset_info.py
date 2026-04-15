# Print out the dataset information for the Enron dataset, including the 
# number of emails, the number of users, and the average number of emails
# per user. This will help us understand the dataset better and plan our
# preprocessing steps accordingly.

# Load the dataset from the enron_emails_labeled.csv file and print out the required information.
import pandas as pd

def print_dataset_info(file_path: str):
    df = pd.read_csv(file_path)
    
    num_emails = len(df)
    unique_users = set(df['from'].unique())
    unique_users.update(df['to'].unique())
    num_users = len(unique_users)
    avg_emails_per_user = num_emails / num_users if num_users > 0 else 0
    
    print(f"Number of emails: {num_emails}")
    print(f"Number of users: {num_users}")
    print(f"Average number of emails per user: {avg_emails_per_user:.2f}")

    # Print the total number of privileged and non-privileged emails
    num_privileged = df['label'].sum()
    num_non_privileged = num_emails - num_privileged
    print(f"Number of privileged emails: {num_privileged}")
    print(f"Number of non-privileged emails: {num_non_privileged}")
    print(f"Percentage of privileged emails: {num_privileged / num_emails * 100:.2f}%")

    print("Columns : ", df.columns.tolist())

print_dataset_info("enron_emails_labeled.csv")
