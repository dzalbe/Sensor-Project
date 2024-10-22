from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def calculate_time_range(days=0, weeks=0, months=0):
    end_timestamp = datetime.utcnow()
    start_timestamp = end_timestamp
    if days > 0:
        start_timestamp -= timedelta(days=days)
    if weeks > 0:
        start_timestamp -= timedelta(weeks=weeks)
    if months > 0:
        start_timestamp -= relativedelta(months=months)
    return start_timestamp, end_timestamp
