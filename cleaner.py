import os
import shutil
from datetime import datetime
import extract_msg

# Define the folder paths
source_folder = r'C:\Users\JGH\Documents\email_test'  # Source folder containing the emails
old_emails_folder = r'C:\Users\JGH\Documents\email_test_old'  # Destination folder for old emails

# Ensure the destination folder exists
os.makedirs(old_emails_folder, exist_ok=True)

# Define the cutoff date (offset-naive)
cutoff_date = datetime(2024, 1, 1)

# Iterate through all files in the source folder
for file_name in os.listdir(source_folder):
    file_path = os.path.join(source_folder, file_name)

    # Skip if it's not a .msg file
    if not file_name.endswith('.msg'):
        continue

    try:
        # Extract the email metadata
        msg = extract_msg.Message(file_path)
        msg_date = msg.date

        # Ensure the date is a datetime object
        if isinstance(msg_date, str):
            email_date = datetime.strptime(msg_date[:16], "%a, %d %b %Y")
        elif isinstance(msg_date, datetime):
            # Remove timezone info to make it offset-naive
            email_date = msg_date.replace(tzinfo=None)
        else:
            print(f"Unknown date format for {file_name}: {msg_date}")
            msg.close()  # Ensure the file is closed
            continue

        # Check if the email is older than the cutoff date
        if email_date < cutoff_date:
            # Move the file to the "old emails" folder
            dest_path = os.path.join(old_emails_folder, file_name)
            shutil.move(file_path, dest_path)
            print(f"Moved: {file_name} (Sent on {email_date})")
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")
    finally:
        # Close the message file handle to prevent locking
        try:
            msg.close()
        except Exception as e:
            print(f"Failed to close {file_name}: {e}")
