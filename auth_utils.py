import streamlit as st
import hashlib
import sqlite3
import uuid
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            subscription_tier TEXT DEFAULT 'free',
            predictions_used INTEGER DEFAULT 0,
            reset_token TEXT,
            reset_token_expires TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            file_name TEXT,
            prediction_days INTEGER,
            predicted_balance REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    if salt is None:
        salt = str(uuid.uuid4())
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}${hash_obj.hex()}", salt

def verify_password(password, stored_hash):
    try:
        salt, _ = stored_hash.split('$')
        new_hash, _ = hash_password(password, salt)
        return new_hash == stored_hash
    except:
        return False

def generate_token():
    return str(uuid.uuid4()) + str(int(time.time()))

def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    if user:
        return {
            "id": user[0], "email": user[1], "password_hash": user[2],
            "created_at": user[3], "last_login": user[4], "is_active": user[5],
            "subscription_tier": user[6], "predictions_used": user[7]
        }
    return None

def create_user(email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        user_id = str(uuid.uuid4())
        password_hash, _ = hash_password(password)
        c.execute("""
            INSERT INTO users (id, email, password_hash, subscription_tier)
            VALUES (?, ?, ?, 'free')
        """, (user_id, email, password_hash))
        conn.commit()
        conn.close()
        return True, user_id
    except sqlite3.IntegrityError:
        return False, "Email already exists"

def verify_login(email, password):
    user = get_user_by_email(email)
    if user and user["is_active"]:
        if verify_password(password, user["password_hash"]):
            update_last_login(user["id"])
            return True, user
        return False, "Invalid password"
    return False, "User not found"

def update_last_login(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def create_reset_token(email):
    user = get_user_by_email(email)
    if user:
        token = generate_token()
        expires = datetime.now() + timedelta(hours=1)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE users SET reset_token = ?, reset_token_expires = ?
            WHERE email = ?
        """, (token, expires, email))
        conn.commit()
        conn.close()
        return token, expires
    return None, None

def verify_reset_token(token):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id FROM users
        WHERE reset_token = ? AND reset_token_expires > ?
    """, (token, datetime.now()))
    user = c.fetchone()
    conn.close()
    return user[0] if user else None

def reset_password(token, new_password):
    user_id = verify_reset_token(token)
    if user_id:
        password_hash, _ = hash_password(new_password)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE users SET password_hash = ?, reset_token = NULL, reset_token_expires = NULL
            WHERE id = ?
        """, (password_hash, user_id))
        conn.commit()
        conn.close()
        return True
    return False

def update_prediction_count(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE users SET predictions_used = predictions_used + 1
        WHERE id = ?
    """, (user_id,))
    conn.commit()
    conn.close()

def get_user_stats(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT predictions_used, subscription_tier FROM users WHERE id = ?", (user_id,))
    stats = c.fetchone()
    c.execute("SELECT COUNT(*) FROM prediction_logs WHERE user_id = ?", (user_id,))
    logs = c.fetchone()[0]
    conn.close()
    return {"predictions_used": stats[0], "subscription_tier": stats[1], "total_predictions": logs}

def send_reset_email(email, token, smtp_config=None):
    if not smtp_config:
        return False, "Email not configured"
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['sender']
        msg['To'] = email
        msg['Subject'] = "Password Reset - CashFlow Predictor"
        body = f"""
        <html>
        <body>
            <h2>Password Reset Request</h2>
            <p>Click the link below to reset your password:</p>
            <p><a href="{smtp_config['reset_url']}?token={token}">Reset Password</a></p>
            <p>This link expires in 1 hour.</p>
            <p>If you didn't request this, ignore this email.</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
            server.starttls()
            server.login(smtp_config['sender'], smtp_config['password'])
            server.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

def change_subscription(user_id, tier):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET subscription_tier = ? WHERE id = ?", (tier, user_id))
    conn.commit()
    conn.close()

def delete_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM prediction_logs WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

init_db()