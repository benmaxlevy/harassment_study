import sys

import pandas as pd

import mh_classifier

from joblib import dump, load


def getParticipantMessages(df):
    # construct list of participants' usernames
    profiles = pd.read_csv("../../IGDD-Dump/Profile.txt", sep="\t")
    usernames = profiles["Username"].dropna().tolist()

    return df.loc[df["Username"].isin(usernames)]


def main():
    # anxiety-related
    anxiety_df = pd.read_csv("reddit_data/mental_health/anxiety.csv", index_col=0)
    panic1_df = pd.read_csv("reddit_data/mental_health/panicdisorder.csv", index_col=0)
    panic2_df = pd.read_csv("reddit_data/mental_health/panicattack.csv", index_col=0)

    # suicide-related
    suicide_df = pd.read_csv("reddit_data/mental_health/suicidewatch.csv", index_col=0)

    # depression-related
    depression1_df = pd.read_csv("reddit_data/mental_health/depression.csv", index_col=0)
    depression2_df = pd.read_csv("reddit_data/mental_health/depression_help.csv", index_col=0)

    # stress-related
    stress_df = pd.read_csv("reddit_data/mental_health/stress.csv", index_col=0)

    # control-related
    aww_df = pd.read_csv("reddit_data/controls/aww.csv", index_col=0)
    mi_df = pd.read_csv("reddit_data/controls/mildlyinteresting.csv", index_col=0)
    st_df = pd.read_csv("reddit_data/controls/showerthoughts.csv", index_col=0)

    mh_df = pd.concat([suicide_df, aww_df, mi_df, st_df, panic1_df, panic2_df, depression1_df, depression2_df, anxiety_df, stress_df]).sample(frac=1)
    mh_df = mh_df[mh_df["text"] != "[removed]"]
    mh_df = mh_df[mh_df["text"] != "removed"]

    try:
        if sys.argv[1] == "existing_model":
            clf = load("model.joblib")
    except:
        clf = mh_classifier.MHClassifier(mh_df, samples=False, ngram_range=(1, 2), word2vec=True)
        # fit model (and print metrics)
        #print(clf.logistic())
        print(clf.mlp())
        # print(clf.multinomialNB())
        # print(clf.rf())
        # print(clf.knn(n=5))
        # print(clf.svm())
        dump(clf.clf, "model.joblib")

    # all messages within igdd dataset
    df = pd.read_csv("../../IGDD-Dump/harassment.txt", sep="\t")
    df = pd.concat([pd.read_csv("../../IGDD-Dump/hate speech.txt", sep="\t"),
                    pd.read_csv("../../IGDD-Dump/nudity_porn.txt", sep="\t"),
                    pd.read_csv("../../IGDD-Dump/sale or promotion of illegal activities.txt", sep="\t"),
                    pd.read_csv("../../IGDD-Dump/self-injury.txt", sep="\t"),
                    pd.read_csv("../../IGDD-Dump/sexual messages_solicitation.txt", sep="\t"),
                    pd.read_csv("../../IGDD-Dump/violence_threat of violence.txt", sep="\t")])
    df = df.dropna(subset=["Message"])

    df_responses = getParticipantMessages(df)
    # group these messages by user
    grouped_users = df_responses.groupby("Username")["ConversationID"].apply(list).reset_index(name="ConvoID")
    grouped_users["Message"] = df_responses.groupby("Username")["Message"].apply(list).reset_index(name="Message")[
        "Message"]

    grouped_users["Message"] = grouped_users["Message"].str.encode(encoding="utf-8")

    results = []

    for row in grouped_users.iterrows():
        results.append({"username": row[1]["Username"], "messages": row[1]["Message"], "ConvoIDs": row[1]["ConvoID"]})
    print(results)
    mh_responses = clf.clf.predict(clf.vectorize(df_responses, "Message"))



if __name__ == "__main__":
    main()
