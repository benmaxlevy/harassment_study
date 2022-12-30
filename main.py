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

    # all messages within igdd dataset
    df = pd.read_csv("../../IGDD-Dump/harassment.txt", sep="\t")
    df = pd.concat([pd.read_csv("../../IGDD-Dump/hate\ speech.txt", sep="\t"), pd.read_csv("../../IGDD-Dump/nudity_porn.txt", sep="\t"), pd.read_csv("../../IGDD-Dump/sale\ or\ promotion\ of\ illegal\ activities.txt", sep="\t"), pd.read_csv("../../IGDD-Dump/self-injury.txt", sep="\t"), pd.read_csv("../../IGDD-Dump/sexual\ messages_solicitation.txt", sep="\t"), pd.read_csv("../../IGDD-Dump/violence_threat\ of\ violence.txt", sep="\t")])
    df = df.dropna(subset=["Message"])

    df_responses = getParticipantMessages(df)
    # group these messages by user
    grouped_users = df_responses.groupby("Username").agg(list)
    print("".join(grouped_users).encode(errors="replace").decode(errors="replace"))

    mh_responses = clf.clf.predict(clf.vectorize(df_responses, "Message"))



if __name__ == "__main__":
    main()
