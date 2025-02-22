import boto3
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email():
    SENDER = "Reservation lambda <abc@prabodhagarwal.com>"
    RECIPIENT = "prabodh1194@gmail.com"
    SUBJECT = "reservation available"
    BODY_TEXT = "reservation available"

    # The character encoding for the email.
    CHARSET = "utf-8"
    msg = MIMEMultipart("mixed")

    # Add subject, from and to lines.
    msg["Subject"] = SUBJECT
    msg["From"] = SENDER
    msg["To"] = RECIPIENT

    # Create a multipart/alternative child container.
    msg_body = MIMEMultipart("alternative")

    # Encode the text and HTML content and set the character encoding. This step is
    # necessary if you're sending a message with characters outside the ASCII range.
    textpart = MIMEText(BODY_TEXT, "plain", CHARSET)

    # Add the text and HTML parts to the child container.
    msg_body.attach(textpart)

    # Attach the multipart/alternative child container to the multipart/mixed
    # parent container.
    msg.attach(msg_body)

    # changes start from here
    strmsg = str(msg)
    body = bytes(strmsg, CHARSET)

    client = boto3.client("sesv2")
    response = client.send_email(
        FromEmailAddress=SENDER,
        Destination={"ToAddresses": [RECIPIENT]},
        Content={"Raw": {"Data": body}},
    )
    print(response)


send_email()
