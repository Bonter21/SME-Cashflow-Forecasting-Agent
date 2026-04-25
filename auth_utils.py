"""
Supabase Authentication for CashFlow Predictor Pro
"""
import streamlit as st
from supabase import create_client, Client
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

PRICING_TIERS = {
    'free': {'predictions': 10, 'pdf': False, 'arima': False, 'email': False},
    'pro': {'predictions': 100, 'pdf': True, 'arima': True, 'email': False, 'price': 9.99},
    'enterprise': {'predictions': -1, 'pdf': True, 'arima': True, 'email': True, 'price': 49}
}

def get_supabase_client():
    try:
        if hasattr(st, 'secrets') and 'SUPABASE_URL' in st.secrets:
            supabase_url = st.secrets['SUPABASE_URL']
            supabase_key = st.secrets['SUPABASE_KEY']
            return create_client(supabase_url, supabase_key)
    except:
        pass
    return None

# ============ USER FUNCTIONS ============

def get_user_by_email(email):
    supabase = get_supabase_client()
    if not supabase:
        return None
    try:
        response = supabase.table('users').select('*').eq('email', email).execute()
        if response.data:
            return response.data[0]
    except:
        pass
    return None

def create_user(email, password):
    supabase = get_supabase_client()
    if not supabase:
        return False, "Supabase not configured"
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        if response.user:
            user_data = {
                "id": response.user.id,
                "email": email,
                "subscription_tier": "free",
                "predictions_used": 0,
                "created_at": datetime.now().isoformat()
            }
            supabase.table('users').insert(user_data).execute()
            return True, response.user.id
        return False, "Signup failed"
    except Exception as e:
        return False, str(e)

def verify_login(email, password):
    supabase = get_supabase_client()
    if not supabase:
        return False, "Supabase not configured"
    try:
        response = supabase.auth.sign_in_with_credentials({
            "email": email,
            "password": password
        })
        if response.user:
            user = get_user_by_email(email)
            return True, user
        return False, "Login failed"
    except Exception as e:
        return False, "Invalid credentials"

def verify_reset_token(token):
    return None

def create_reset_token(email):
    user = get_user_by_email(email)
    if user:
        token = f"reset_{user['id']}"
        return token, datetime.now() + timedelta(hours=1)
    return None, None

def reset_password(token, new_password):
    return False

def send_reset_email(email, token, smtp_config=None):
    if not smtp_config or not smtp_config.get('server'):
        return False, "Email not configured"
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['sender']
        msg['To'] = email
        msg['Subject'] = "Password Reset - CashFlow Predictor"
        body = f"""
        <html><body>
            <h2>Password Reset</h2>
            <p>Click to reset: <a href="{smtp_config['reset_url']}?token={token}">Reset Password</a></p>
            <p>Expires in 1 hour.</p>
        </body></html>
        """
        msg.attach(MIMEText(body, 'html'))
        with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
            server.starttls()
            server.login(smtp_config['sender'], smtp_config['password'])
            server.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

def get_user_stats(user_id):
    supabase = get_supabase_client()
    if not supabase:
        return {"predictions_used": 0, "subscription_tier": "free", "total_predictions": 0}
    try:
        response = supabase.table('users').select('*').eq('id', user_id).execute()
        if response.data:
            user = response.data[0]
            return {
                "predictions_used": user.get('predictions_used', 0),
                "subscription_tier": user.get('subscription_tier', 'free'),
                "total_predictions": user.get('predictions_used', 0)
            }
    except:
        pass
    return {"predictions_used": 0, "subscription_tier": "free", "total_predictions": 0}

def update_prediction_count(user_id):
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        stats = get_user_stats(user_id)
        new_count = stats['predictions_used'] + 1
        supabase.table('users').update({'predictions_used': new_count}).eq('id', user_id).execute()
    except:
        pass

def delete_user(user_id):
    supabase = get_supabase_client()
    if supabase:
        try:
            supabase.table('users').delete().eq('id', user_id).execute()
        except:
            pass

def change_subscription(user_id, tier):
    supabase = get_supabase_client()
    if supabase:
        try:
            supabase.table('users').update({'subscription_tier': tier}).eq('id', user_id).execute()
        except:
            pass

# ============ PREDICTION HISTORY FUNCTIONS ============

def save_prediction(user_id, file_name, prediction_data, conclusion):
    """Save a prediction to history"""
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        prediction_record = {
            "user_id": user_id,
            "file_name": file_name,
            "prediction_days": prediction_data.get('days', 30),
            "current_balance": prediction_data.get('current_balance', 0),
            "predicted_balance": prediction_data.get('predicted_balance', 0),
            "trend": prediction_data.get('trend', 'stable'),
            "status": conclusion.get('status', 'neutral'),
            "conclusion": conclusion.get('title', ''),
            "created_at": datetime.now().isoformat()
        }
        supabase.table('prediction_logs').insert(prediction_record).execute()
        
        # Update user prediction count
        update_prediction_count(user_id)
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def get_prediction_history(user_id, limit=20):
    """Get user's prediction history"""
    supabase = get_supabase_client()
    if not supabase:
        return []
    
    try:
        response = supabase.table('prediction_logs').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(limit).execute()
        return response.data if response.data else []
    except:
        return []

def get_prediction_by_id(log_id, user_id):
    """Get a specific prediction by ID"""
    supabase = get_supabase_client()
    if not supabase:
        return None
    
    try:
        response = supabase.table('prediction_logs').select('*').eq('id', log_id).eq('user_id', user_id).execute()
        return response.data[0] if response.data else None
    except:
        return None

def delete_prediction(log_id, user_id):
    """Delete a prediction from history"""
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        supabase.table('prediction_logs').delete().eq('id', log_id).eq('user_id', user_id).execute()
        return True
    except:
        return False