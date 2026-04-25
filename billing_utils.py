"""
Stripe Billing Integration for CashFlow Predictor Pro
"""
import streamlit as st
import stripe
from datetime import datetime
import urllib.parse

STRIPE_PRICING = {
    'free': {
        'name': 'Free',
        'price': 0,
        'predictions': 10,
        'features': ['Basic predictions', '10/month'],
    },
    'pro': {
        'name': 'Pro',
        'price_id': 'price_pro_monthly',  # Stripe Price ID
        'price': 9.99,
        'predictions': 100,
        'features': ['Unlimited predictions', 'PDF reports', 'ARIMA models', 'Email support'],
    },
    'enterprise': {
        'name': 'Enterprise',
        'price_id': 'price_enterprise_monthly',
        'price': 49.00,
        'predictions': -1,
        'features': ['Everything in Pro', 'API access', 'White-label', 'Priority support', 'Custom integrations'],
    }
}

def get_stripe_client():
    try:
        if hasattr(st, 'secrets') and 'STRIPE_KEY' in st.secrets:
            stripe.api_key = st.secrets['STRIPE_KEY']
            return stripe
    except:
        pass
    return None

def check_subscription(user_id):
    """Check user's current subscription status"""
    from auth_utils import get_user_stats, get_supabase_client
    
    stats = get_user_stats(user_id)
    tier = stats.get('subscription_tier', 'free')
    
    if tier == 'enterprise':
        return {'tier': 'enterprise', 'active': True, 'predictions': -1}
    
    if tier == 'pro':
        return {'tier': 'pro', 'active': True, 'predictions': 100}
    
    return {'tier': 'free', 'active': True, 'predictions': 10}

def can_use_feature(user_id, feature):
    """Check if user can use a specific feature"""
    sub = check_subscription(user_id)
    tier = sub['tier']
    
    pro_features = ['pdf_export', 'arima', 'email_alerts', 'unlimited_predictions', 'api_access']
    enterprise_features = pro_features + ['white_label', 'custom_integrations']
    
    if tier == 'enterprise':
        return True
    if tier == 'pro':
        return feature in pro_features
    return feature in []

def check_prediction_limit(user_id):
    """Check if user has reached prediction limit"""
    from auth_utils import get_user_stats
    
    sub = check_subscription(user_id)
    stats = get_user_stats(user_id)
    
    used = stats.get('predictions_used', 0)
    limit = sub['predictions']
    
    if limit == -1:
        return True, None
    
    remaining = limit - used
    if remaining > 0:
        return True, remaining
    return False, 0

def create_checkout_session(user_id, tier):
    """Create Stripe checkout session"""
    stripe = get_stripe_client()
    if not stripe:
        return None, "Stripe not configured"
    
    try:
        pricing = STRIPE_PRICING.get(tier)
        if not pricing:
            return None, "Invalid tier"
        
        base_url = os.getenv('APP_URL', 'https://your-app.streamlit.app')
        
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': f"CashFlow Predictor Pro - {pricing['name']}",
                        'description': f"{pricing['name']} plan for CashFlow Predictor Pro",
                    },
                    'unit_amount': int(pricing['price'] * 100),
                    'recurring': {'interval': 'month'},
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{base_url}?tier={tier}&success=true",
            cancel_url=f"{base_url}?tier={tier}&canceled=true",
            metadata={'user_id': user_id, 'tier': tier}
        )
        
        return session.url, None
    except Exception as e:
        return None, str(e)

def create_portal_session(user_id):
    """Create Stripe customer portal session"""
    stripe = get_stripe_client()
    if not stripe:
        return None, "Stripe not configured"
    
    try:
        base_url = os.getenv('APP_URL', 'https://your-app.streamlit.app')
        
        session = stripe.billing_portal.Session.create(
            customer='cus_demo',  # Would need to store Stripe customer ID
            return_url=base_url
        )
        
        return session.url, None
    except Exception as e:
        return None, str(e)

def get_subscription_status(stripe_customer_id):
    """Get subscription status from Stripe"""
    stripe = get_stripe_client()
    if not stripe or not stripe_customer_id:
        return 'inactive'
    
    try:
        subscriptions = stripe.Subscription.list(customer=stripe_customer_id, limit=1)
        if subscriptions.data:
            sub = subscriptions.data[0]
            if sub.status == 'active':
                return 'active'
            elif sub.status == 'past_due':
                return 'past_due'
            elif sub.status == 'canceled':
                return 'canceled'
    except:
        pass
    return 'inactive'

def cancel_subscription(stripe_subscription_id):
    """Cancel a subscription"""
    stripe = get_stripe_client()
    if not stripe:
        return False, "Stripe not configured"
    
    try:
        stripe.Subscription.cancel(stripe_subscription_id)
        return True, "Subscription canceled"
    except Exception as e:
        return False, str(e)

def handle_webhook(request):
    """Handle Stripe webhook events"""
    stripe = get_stripe_client()
    if not stripe:
        return False
    
    payload = request.data
    event_type = payload.get('type')
    
    if event_type == 'checkout.session.completed':
        session = payload.get('data', {}).get('object', {})
        user_id = session.get('metadata', {}).get('user_id')
        tier = session.get('metadata', {}).get('tier')
        
        if user_id and tier:
            from auth_utils import change_subscription
            change_subscription(user_id, tier)
            return True
    
    elif event_type == 'customer.subscription.deleted':
        subscription = payload.get('data', {}).get('object', {})
        # Handle cancellation
        return True
    
    return False

# Feature gating decorators
def require_pro(f):
    """Decorator to require Pro subscription"""
    def wrapper(*args, **kwargs):
        user = st.session_state.get('user')
        if not user or not can_use_feature(user.get('id'), 'unlimited_predictions'):
            st.error("🔒 This feature requires Pro subscription. Upgrade to unlock!")
            st.session_state.page = 'profile'
            st.rerun()
        return f(*args, **kwargs)
    return wrapper

def require_enterprise(f):
    """Decorator to require Enterprise subscription"""
    def wrapper(*args, **kwargs):
        user = st.session_state.get('user')
        if not user or not can_use_feature(user.get('id'), 'api_access'):
            st.error("🔒 This feature requires Enterprise subscription. Contact sales!")
            st.session_state.page = 'profile'
            st.rerun()
        return f(*args, **kwargs)
    return wrapper