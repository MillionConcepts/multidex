import datetime as dt

def debug_check_cache(n_intervals, *, cget):
    print(dt.datetime.now().isoformat(), cget('queryset'))
    return 0
