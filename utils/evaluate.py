import pandas as pd

def get_edges_sensitivity(A: pd.DataFrame,
                          events: pd.DataFrame,
                          window: pd.Timedelta,
                          min_events: int = 3
                          ) -> pd.DataFrame:
    A_recall = pd.DataFrame(0., index = A.index, columns = A.columns)
    for ix in A_recall.index:
        if A.loc[ix].sum() > 0:
            starts = pd.Series(events.index[events[ix].astype(bool)])
            starts.index = starts.values
            starts = starts.groupby(starts.index.to_period('W')).min()
            if starts.size >= min_events:
                events_i = pd.DataFrame(index = starts.values, columns = A.columns)
                for t in starts:
                    events_t = events.loc[t : (t + window), A.columns]
                    events_i.loc[t] = (events_t.sum(0) > 0).astype(int)
                tpr, tpr_rand = events_i.mean(0), events.loc[:, A.columns].mean(0)
                A_recall.loc[ix] = (tpr - tpr_rand) / (1 - tpr_rand)
    return A_recall * A