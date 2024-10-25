"""
Application configuration and settings.
"""

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .congestion-free { color: #28a745; }
    .congestion-light { color: #ffc107; }
    .congestion-moderate { color: #fd7e14; }
    .congestion-heavy { color: #dc3545; }
    .congestion-severe { color: #721c24; }
</style>
"""

ASTANA_CENTER = [51.1694, 71.4491]

AVAILABLE_DATASETS = {
    "Small Test": "data/samples/small_test",
    "Astana Week": "data/samples/astana_week",
    "Left Bank Day": "data/samples/left_bank_day",
    "Right Bank Month": "data/samples/right_bank_month",
    "Highways Week": "data/samples/highways_week"
}

DISTRICT_INSIGHTS = {
    'City Center': "Historic center with government buildings. High congestion during business hours due to administrative activities.",
    'Left Bank': "Modern business district with Bayterek Tower. Heavy traffic during rush hours, many shopping centers.",
    'Right Bank': "Residential and commercial area. Moderate traffic, industrial zones create specific congestion patterns.",
    'Esil District': "Government quarter with presidential palace. Restricted access areas affect traffic flow patterns.",
    'Suburban': "Developing residential areas. Traffic mainly from commuters, lighter congestion overall."
}

CONGESTION_LABELS = ["Free Flow", "Light", "Moderate", "Heavy", "Severe"]
CONGESTION_COLORS = ['green', 'yellow', 'orange', 'red', 'darkred']
CONGESTION_HEX_COLORS = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#721c24']

PAGE_CONFIG = {
    "page_title": "Astana Traffic Prediction",
    "page_icon": "Car",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
