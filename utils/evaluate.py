import pandas as pd

def get_sensitivity(window: pd.Timedelta,
                    symbols: pd.Series,
                    events: pd.DataFrame,
                    min_events: int = 3,
                    ) -> pd.DataFrame:
    recall = pd.DataFrame(index = symbols, columns = symbols)
    for ix in symbols:
        starts = pd.Series(events.index[events[ix].astype(bool)])
        starts.index = starts.values
        starts = starts.groupby(starts.index.to_period('W')).min()
        if starts.size >= min_events:
            events_i = pd.DataFrame(index = starts.values, columns = symbols)
            for t in starts:
                events_t = events.loc[t : (t + window), symbols]
                events_i.loc[t] = (events_t.sum(0) > 0).astype(int)
            tpr = events_i.mean(0)
            tpr_rand = events.loc[:, symbols].mean(0)
            recall.loc[ix] = (tpr - tpr_rand) / (1 - tpr_rand)
    return recall