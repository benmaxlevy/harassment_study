import sys
import os
import requests
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import PreProcess


def collect(subreddit, limit, cycles, assumed_class, title):
    df = pd.DataFrame(columns=["text", "label"])

    column = "selftext"
    if title.lower() == "true":
        column = "title"

    before = ""

    for i in range(int(cycles)):
        print(before)
        try:
            data = json.loads(requests.get(f"http://api.pushshift.io/reddit/submission/search?subreddit={subreddit}&size={limit}&before={before}",
                                       headers={'User-agent': 'collector 1.0'}, timeout=120).text)["data"]
        except:
            print("Error occurred while parsing the JSON — continuing.")
            continue

        for post in data:
            post["selftext"] = post[column].encode("ascii", "ignore").decode("ascii")
            if post[column] != "":
                df.loc[len(df.index)] = [post[column], assumed_class]

        if len(data) == 0:
            # no more data!
            break
        else:
            before = data[len(data)-1]["created_utc"]

    return df


if __name__ == "__main__":
    if len(sys.argv) == 6:
        # second arg is the subreddit, third arg is the number of posts to gather, fourth=number of cycles (number of posts collected will be (third arg)*(fourth arg), fifth=class to set, sixth=title or post (True=title, False=post)
        df = collect(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        preprocessor = PreProcess(df, "text")
        preprocessor.lemmatize()
        df = preprocessor.returnDf()

        # save csv to file named f"{sys.argv[1]}.csv"
        dataset_file = open(f"reddit_data/{sys.argv[1][sys.argv[1].index('/')+1:]}.csv", "w")
        dataset_file.write(df.to_csv())

    else:
        raise ValueError("Expected one parameter specifying the subreddit to collect from, one parameter for the " +
                         "number of posts to gather, one parameter for the number of cycles, and one for the assumed "
                         "label (e.g., suicide).")
