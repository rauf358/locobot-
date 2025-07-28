import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv, find_dotenv # Import find_dotenv

# --- DIAGNOSTIC STEP ---
# Try to explicitly find the .env file
dotenv_path = find_dotenv()
if dotenv_path:
    print(f"Found .env file at: {dotenv_path}")
    # Load environment variables from the found .env file
    load_dotenv(dotenv_path)
else:
    print("WARNING: .env file not found. Environment variables might not be loaded.")
# --- END DIAGNOSTIC STEP ---


SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD") # Or os.getenv("SENDER_PASSWORD") if you standardize

RECEIVER_EMAIL ="detective2k18@gmail.com" # Or any other email you want to test sending to
TEST_SUBJECT = "Test Email from Python Script"
TEST_BODY = "This is a test email sent from your Python script using smtplib."

print(f"Attempting to send email from: {SENDER_EMAIL}")
print(f"To: {RECEIVER_EMAIL}")
print(f"SMTP Server: {SMTP_SERVER}:{SMTP_PORT}")
print(f"SENDER_PASSWORD (first 3 chars): {SENDER_PASSWORD[:3] if SENDER_PASSWORD else 'None'}") # Print partial password for debug


msg = MIMEMultipart()
msg['From'] = SENDER_EMAIL
msg['To'] = RECEIVER_EMAIL
msg['Subject'] = TEST_SUBJECT
msg.attach(MIMEText(TEST_BODY, 'plain'))

try:
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
    print("Email sent successfully!")
except smtplib.SMTPAuthenticationError as e:
    print(f"ERROR: SMTP Authentication Failed. Details: {e}")
    print("Possible causes:")
    print("1. Incorrect SENDER_EMAIL or SENDER_PASSWORD in .env.")
    print("2. 'App Password' required for Gmail (if 2-Step Verification is ON).")
    print("   (Go to https://myaccount.google.com/apppasswords while logged into the sender account)")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

