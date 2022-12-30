import pandas as pd

import mh_classifier


def getParticipantMessages(df):
    # construct list of participants' usernames
    profiles = pd.read_csv("../../IGDD-Dump/Profile.txt", sep="\t")
    usernames = profiles["Username"].dropna().tolist()

    return df.loc[df["Username"].isin(usernames)]


def main():
    anxiety_df = pd.read_csv("reddit_data/mental_health/anxiety.csv", index_col=0)
    suicide_df = pd.read_csv("reddit_data/mental_health/suicidewatch.csv", index_col=0)
    depression_df = pd.read_csv("reddit_data/mental_health/depression.csv", index_col=0)
    panic_df = pd.read_csv("reddit_data/mental_health/panicdisorder.csv", index_col=0)
    aww_df = pd.read_csv("reddit_data/controls/aww.csv", index_col=0)
    mi_df = pd.read_csv("reddit_data/controls/mildlyinteresting.csv", index_col=0)
    st_df = pd.read_csv("reddit_data/controls/showerthoughts.csv", index_col=0)

    mh_df = pd.concat([suicide_df, aww_df, mi_df, st_df, depression_df, anxiety_df]).sample(frac=1)
    mh_df = mh_df[mh_df["text"] != "[removed]"]

    """ uncomment if the controls are in a sep. file 
    
    control_df = pd.read_csv("reddit_data/control.csv")

    mh_df = pd.concat([mh_df, control_df]) 
    """

    clf = mh_classifier.MHClassifier(mh_df, samples=False, ngram_range=(1, 2), word2vec=True)
    # fit model (and print metrics)
    #print(clf.logistic())
    print(clf.mlp())
    # print(clf.multinomialNB())
    # print(clf.rf())
    # print(clf.knn(n=5))
    # print(clf.svm())

    # find ratio and % of harassment messages that indicate suicide intents 
    df = pd.read_csv("../../IGDD-Dump/harassment.txt", sep="\t")
    df = df.dropna(subset=["Message"])

    df_responses = getParticipantMessages(df)
    print(len(df_responses))
    # suicide

    suicide_results = clf.clf.predict(clf.vectorize(df_responses, "Message"))

    count = 0

    suicide = 0

    for result in suicide_results:
        if result == "suicide":
            suicide = suicide + 1



if __name__ == "__main__":
    main()
