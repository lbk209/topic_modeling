import collections
import os, re
import pandas as pd
import numpy as np
from tqdm import tqdm

from typing import List, Union

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as color_qual
import plotly.io as pio

from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest

SENTIMENT_LABELS = ['positive', 'neutral', 'negative']

def read_csv(file, path_data, cols_eval=None, **kwargs):
    """
    kwargs: keyword args for pd.read_csv
    """
    files = [x for x in os.listdir(path_data) if x.startswith(file)]

    if len(files) == 0:
        print('ERROR!: No csv to read')
        return None

    if cols_eval is None:
        converters=None
    else:
        if not isinstance(cols_eval, list):
            cols_eval = [cols_eval]
        converters= {c: lambda x: eval(x) for c in cols_eval}

    df_reviews = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f'{path_data}/{f}', converters=converters, **kwargs)
        df_reviews = pd.concat([df_reviews, df])

    return df_reviews.reset_index(drop=True)


def split_str(string, length=50, split='\n', indent=''):
    words = string.split()

    words_split = ''
    current_length = 0
    for i, word in enumerate(words):
        if current_length + len(word) <= length:
            words_split += f'{word} '
            current_length += len(word) + 1  # +1 for the space
        else:
            words_split += f'{split}{indent}{word} '
            current_length = len(word) + len(indent) + 1
    return words_split


def print_with_line_feed(input_string, line_length=50, split='\n', indent='  '):
    ws = split_str(input_string, length=line_length, split=split, indent=indent)
    print(ws)


def check_topic_aspect(df_topic_info, aspect, aspect_default='Representation', start_idx=3, warning=True):
    """
    df_topic_info: BERTopic.get_topic_info()
    """
    # exclude 1st 3 cols Topic, Count and Name. CustomName could be included which is not list
    aspect_list = list(df_topic_info.columns)[start_idx:]
    if aspect not in aspect_list:
        if warning:
            print(f'WARNING: aspect {aspect} is not in {aspect_list}')
        if aspect_default not in aspect_list:
            print(f'ERROR: even default aspect {aspect_default} is not in {aspect_list}')
            aspect = None
        else:
            print(f'Default aspect {aspect_default} is used.')
            aspect = aspect_default
    return aspect


class utils():
    def __init__(self, topic_model, reduced_embeddings=None, docs=None):
        self.topic_model = topic_model
        # for visualize_documents
        self.reduced_embeddings = reduced_embeddings
        self.docs = docs
        self.count_children = 0


    def _check_var(self, var_arg, var_self):
        if var_arg is None:
            var_arg = var_self
        return var_arg


    def print_topic_info(self):
        """
        print number of topics and percentage of outliers
        """
        df = self.topic_model.get_topic_info()

        a = len(df.loc[df.Topic > -1])
        print(f'num of topics: {a}')

        a = df.loc[df.Topic == -1]['Count']
        if a.count() > 0:
            a = a.values[0]/df['Count'].sum()
        else:
            a = 0
        print(f'outliers: {a:.3f}')

        return df


    def get_topic_labels(self, list_tid=None, aspect=None, min_count=0, length=120, sort_by_input=True, print_labels=True):
        """
        list_tid: list of topics
        length: number of chars to print every line
        sort_by_input: sort dict by the order of topic in list_tid
        dict_label: dict of topic and custom label
        """
        aspect = self._check_aspect(aspect)
        if aspect is None:
            return None

        df = self.topic_model.get_topic_info()
        cond = (df.Count > min_count)
        if list_tid is not None:
            if not isinstance(list_tid, list):
                list_tid = [list_tid]
            cond = cond & (df.Topic.isin(list_tid))

        df = df.loc[cond]
        if sort_by_input and (list_tid is not None):
            df = df.sort_values(by='Topic', key=lambda x: x.map({v: i for i, v in enumerate(list_tid)}))

        dict_label = df.set_index('Topic')[aspect].to_dict()
        dict_label = {k: ', '.join(v) if isinstance(v, (list, tuple)) else v for k, v in dict_label.items()}

        if print_labels:
            _ = [print_with_line_feed(f'{k}: {v}', length) for k,v in dict_label.items()]

        return dict_label


    def _check_aspect(self, aspect=None, aspect_default='Representation'):
        df_topic_info = self.topic_model.get_topic_info()
        return check_topic_aspect(df_topic_info, aspect, aspect_default=aspect_default,
                                  start_idx=3, warning=False)


    def get_representative_docs(self, list_tid=None, length=120, max_topics=5):
        rep_docs = self.topic_model.get_representative_docs()

        if list_tid is not None:
            rep_docs = {k: v for k, v in rep_docs.items() if k in list_tid}

        for i, (k, v) in enumerate(rep_docs.items()):
            _ = [print_with_line_feed(f'{k}-{x}: {y}', length) for x, y in enumerate(v)]
            if i > max_topics:
                print(f'the docs of {max_topics} topics printed.')
                break

        return rep_docs


    def get_topic_docs(self, topic, class_name=None, num_print=5, length_print=120,
                       docs=None, classes_all=None, aspect=None):
        """
        get docs of the topic and print the num_print of docs.
        """
        docs = self._check_var(docs, self.docs)
        if docs is None:
            print('ERROR!: missing docs')
            return None

        topics = self.topic_model.topics_
        if (class_name is not None) and (classes_all is not None):
            cond = lambda x: (x[0] == topic) and (class_name.lower() in x[1].lower())
            topic_docs = [docs[i] for i, x in enumerate(zip(topics, classes_all)) if cond(x)]
        else:
            topic_docs = [docs[i] for i, x in enumerate(topics) if x == topic]

        # print docs of num_print
        if num_print > 0:
            if len(topic_docs) > num_print:
                print(f'Printing {num_print} docs from {len(topic_docs)}')
                docs_print = np.random.choice(topic_docs, num_print, replace=False)
            else:
                docs_print = topic_docs

            topic_to_label = self.get_topic_labels(aspect=aspect, print_labels=False)
            print(f'Topic {topic}: {topic_to_label[topic]}')
            _ = [print_with_line_feed(f'-. {x}', length_print, indent='') for x in docs_print]

        return topic_docs



    def merge_topics(self, topics_to_merge, docs=None, name='KeyBERT'):

        docs = self._check_var(docs, self.docs)
        if docs is None:
            print('ERROR!: docs required to merge topics')
            return None

        self.topic_model.merge_topics(docs, topics_to_merge)
        # update custom labels
        self.set_custom_labels(name=name)


    def get_topics(self, index, num_topics=None, cols = ['Topic', 'KeyBERT']):
        """
        get a row from df, the result of bertopic_batch
        index: index of a param set
        """
        df = self.topic_model.get_topic_info()
        if num_topics is None:
            num_topics = 99999
        # get the position of topic 0 which might be 0 if no outlier
        i = df.loc[df.Topic==0].index[0]
        return df.iloc[i:num_topics+i].loc[:, cols].rename(columns=dict(zip(cols, ['index', index]))).set_index('index').transpose()


    def set_custom_labels(self, name='KeyBERT'):
        #length = 40
        #end = ' ...'

        #labels = {topic: '; '.join(list(zip(*values))[0][:n_words]) + end for topic, values in topic_model.topic_aspects_[name].items()}
        labels = {topic: f'{topic} ' + '; '.join(list(zip(*values))[0]) for topic, values in self.topic_model.topic_aspects_[name].items()}

        #labels = {k: v[:length] + end for k,v in labels.items()}
        #labels = {k: '\n'.join([v[i:i+40] for i in range(0, len(v), 40)]) for k,v in labels.items()}
        self.topic_model.set_topic_labels(labels)

        if self.count_children > 0:
            print('WARNING: create the children instances again such as visualize.')
        #return topic_model


    def check_similarity(self, custom_labels=False,
                         embedding_model=None, min_distance=0.8,
                         pytorch_cos_sim=None):

        """
        pytorch_cos_sim: sentence_transformers.util.pytorch_cos_sim
        """
        topic_model = self.topic_model
        distance_matrix = cosine_similarity(np.array(topic_model.topic_embeddings_))

        if custom_labels:
            list_labels = topic_model.custom_labels_
            sep = ' '
        else:
            list_labels = topic_model.topic_labels_.values()
            sep = '_'

        dist_df = pd.DataFrame(distance_matrix, columns=list_labels, index=list_labels)

        tmp = []
        for rec in dist_df.reset_index().to_dict('records'):
            t1 = rec['index']
            for t2 in rec:
                if t2 == 'index':
                    continue
                tmp.append(
                    {
                        'topic1': t1,
                        'topic2': t2,
                        'distance': rec[t2]
                    }
                )

        pair_dist_df = pd.DataFrame(tmp)

        pair_dist_df = pair_dist_df[(pair_dist_df.topic1.map(lambda x: not x.startswith('-1'))) &
                    (pair_dist_df.topic2.map(lambda x: not x.startswith('-1')))]

        pair_dist_df = (pair_dist_df[pair_dist_df.topic1 < pair_dist_df.topic2]
                .sort_values('distance', ascending = False)
                .reset_index(drop=True))

        if (embedding_model is not None) and (pytorch_cos_sim is not None):
            print(f'Calculating the similarity of custom label pairs for which the topic similarity exceeds {min_distance}...')
            encode = lambda x: embedding_model.encode(x, convert_to_tensor=True)
            pair_dist_df = pair_dist_df.join(pair_dist_df
                                             .loc[pair_dist_df.distance >= min_distance]
                                             .apply(lambda x: pytorch_cos_sim(encode(x.topic1), encode(x.topic2))[0][0].item(), axis=1)
                                             .rename('c/label sim')
                                             , how='right')

        # add each topic pair as a set for convenient indexing
        pair_dist_df['pair'] = pair_dist_df.apply(lambda x: set(int(x.iloc[i].split(sep)[0]) for i in range(2)), axis=1)
        return pair_dist_df


    def calc_score(self, aspect='KeyBERT', tid=None):
        """
        extract c-TF-IDF scores of words in aspect KeyBERT
        and calc mean, median and std for each topic
        """
        topic_model = self.topic_model

        if aspect not in topic_model.topic_aspects_.keys():
            print('ERROR')

        scores_all = topic_model.topic_aspects_[aspect]
        if tid is None:
            scores = {k:v for k,v in scores_all.items() if k > -1}
        else:
            scores = {k:v for k,v in scores_all.items() if k in tid}

        list_mean = []
        list_median = []
        list_std = []
        for t, d in scores.items():
            s = [x[1] for x in d]
            list_mean.append(np.mean(s))
            list_median.append(np.median(s))
            list_std.append(np.std(s))

        return (list_mean, list_median, list_std)


    def visualize(self, docs=None, classes=None):
        """
        create a instance of visualize class
        """
        self.count_children += 1
        if docs is None:
            docs = self.docs
        # return a instance of the class visualize
        return visualize(self.topic_model, docs=docs, classes=classes,
                         reduced_embeddings=self.reduced_embeddings)


    def multi_topics_stats(self, docs=None, df_data=None, col_class=None):
        """
        create a instance of multi_topics_stats class
        """
        docs = self._check_var(docs, self.docs)
        if docs is None:
            print('ERROR!: docs required for approximate_distribution')
            return
            
        self.count_children += 1
        topic_model = self.topic_model

        args_distr = topic_model.approximate_distribution(docs, calculate_tokens=True)
        df_topic_info = topic_model.get_topic_info()
        return multi_topics_stats(*args_distr, df_data=df_data, df_topic_info=df_topic_info, col_class=col_class)


    def multi_topics_sentiment(self, sentiment_analysis,
                               max_sequence_length=2400):
        """
        create a instance of multi_topics_sentiment class
        """
        self.count_children += 1
        return multi_topics_sentiment(topic_model=self.topic_model,
                                      sentiment_analysis=sentiment_analysis,
                                      max_sequence_length=max_sequence_length)


class visualize():
    def __init__(self, topic_model, docs=None, classes=None, reduced_embeddings=None, custom_labels=False):
        self.topic_model = topic_model
        self.docs = docs
        self.classes = classes
        self.reduced_embeddings = reduced_embeddings
        self.custom_labels = custom_labels
        self.hierarchical_topics = None

    def _check_flag_cl(self, custom_labels):
        if custom_labels is None:
            custom_labels = self.custom_labels
        return custom_labels

    def _check_var(self, var_arg, var_self):
        if var_arg is None:
            var_arg = var_self
        return var_arg


    def documents(self, docs=None, list_tid=None, custom_labels=None,
                            hide_annotations=True, hide_document_hover=False, **kwargs):
        """
        custom_labels: set to False if custom label is long enough to fill the whole fig
        """
        docs = self._check_var(docs, self.docs)
        if docs is None:
            print('ERROR!: No docs assigned')
            return None
        else:
            docs = [x[:100] for x in docs]

        if list_tid is None:
            list_tid = range(20)

        custom_labels = self._check_flag_cl(custom_labels)

        return self.topic_model.visualize_documents(docs, topics=list_tid, custom_labels=custom_labels,
                                        hide_annotations=hide_annotations, hide_document_hover=hide_document_hover,
                                        reduced_embeddings=self.reduced_embeddings, **kwargs)


    def hierarchy(self, docs=None, **kwargs):
        docs = self._check_var(docs, self.docs)
        if docs is None:
            print('ERROR!: No docs assigned')
            return None

        # Extract hierarchical topics and their representations
        self.hierarchical_topics = self.topic_model.hierarchical_topics(docs)

        # Visualize these representations
        return self.topic_model.visualize_hierarchy(hierarchical_topics=self.hierarchical_topics, **kwargs)


    def barchart(self, **kwargs):
        return self.topic_model.visualize_barchart(**kwargs)


    def topics(self, **kwargs):
        return self.topic_model.visualize_topics(**kwargs)


    def topics_per_multiclass(self,
                         df_class: pd.DataFrame,
                         docs=None,
                         ncols=4, top_n_topics=None,
                         horizontal_spacing=.05,
                         vertical_spacing=.3,
                         width = 350, height = 350):
        """
        grid plot of multi-classes (such as a set of param search)
        df_class: dataframe of classes where columns are class names.
                  must be the same order of docs.
        """
        docs = self._check_var(docs, self.docs)
        if docs is None:
            print('ERROR!: docs required')
            return None

        subplot_titles = [x for x in df_class.columns if not isinstance(x, int)]
        nrows = len(subplot_titles)//ncols+1

        fig = make_subplots(rows=nrows, cols=ncols,
                            shared_xaxes=False,
                            horizontal_spacing=horizontal_spacing,
                            vertical_spacing=vertical_spacing / nrows if nrows > 1 else 0,
                            subplot_titles=subplot_titles)

        row, col = 1, 1
        for i, _ in enumerate(subplot_titles):
            classes = df_class.iloc[:, i].apply(str)
            topics_per_class = self.topic_model.topics_per_class(docs, classes=classes)

            f = self.topic_model.visualize_topics_per_class(topics_per_class,
                                                top_n_topics=top_n_topics,
                                                #width=1000, height=500,
                                                normalize_frequency = False)

            # update visible to show all topics
            _ = [fig.add_trace(x.update({'visible':True}), row=row, col=col) for x in f.data]

            if col == ncols:
                col = 1
                row += 1
            else:
                col += 1

        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            width=width*ncols,
            height=height*nrows if nrows > 1 else height * 1.3,
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Rockwell"
            ),
        )

        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig


    def topics_per_class(self,
                         topics_per_class: pd.DataFrame,
                         group: List[str] = None,
                         docs: List[str] = None,
                         # classes optional
                         classes: List[str] = None,
                         top_n_topics: int = 10,
                         topics: List[int] = None,
                         normalize_frequency: bool = False,
                         relative_share = False,
                         custom_labels = None,
                         title: str = "<b>Topics per Class</b>",
                         width: int = 1250,
                         height: int = 900) -> go.Figure:
        """
        customized BERTopic.visualize_topics_per_class:
         plot relative shares and display only the selected group
        """
        custom_labels = self._check_flag_cl(custom_labels)
        topic_model = self.topic_model
        docs = self._check_var(docs, self.docs)
        colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]

        if group is None:
            freq_df = topic_model.get_topic_freq()
        elif (group is not None) and (docs is None or classes is None):
            print('WARNING: visualizing the selected group needs docs and classes as well\n')
            freq_df = topic_model.get_topic_freq()
        else:
            # redefine topics_per_class for the group
            topics_per_class = topics_per_class.loc[topics_per_class.Class.isin(group)]
            # redefine freq_df for the group
            classes_g = [i for i, x in enumerate(classes) if x in group]
            documents = pd.DataFrame({"Document": docs, "ID": range(len(docs)), "Topic": topic_model.topics_})
            documents = documents.loc[classes_g]
            topic_sizes = collections.Counter(documents.Topic.values.tolist())
            freq_df = (pd
                        .DataFrame(topic_sizes.items(), columns=['Topic', 'Count'])
                        .sort_values("Count",ascending=False)
                        )

        if relative_share:
            if docs is None or classes is None:
                print('WARNING: relative share plot needs docs and classes as well\n')
            else:
                total_freq = pd.DataFrame({"Document": docs, "Class":classes}).groupby('Class')['Document'].count().to_dict()
                topics_per_class = topics_per_class.assign(Frequency=topics_per_class.apply(lambda x: x.Frequency/total_freq[x.Class], axis=1))

        # Select topics based on top_n and topics args
        freq_df = freq_df.loc[freq_df.Topic != -1, :]
        if topics is not None:
            selected_topics = list(topics)
        elif top_n_topics is not None:
            selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
        else:
            selected_topics = sorted(freq_df.Topic.to_list())

        # Prepare data
        if isinstance(custom_labels, str):
            topic_names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
            topic_names = ["_".join([label[0] for label in labels[:4]]) for labels in topic_names]
            topic_names = [label if len(label) < 30 else label[:27] + "..." for label in topic_names]
            topic_names = {key: topic_names[index] for index, key in enumerate(topic_model.topic_labels_.keys())}
        elif topic_model.custom_labels_ is not None and custom_labels:
            topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
        else:
            topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                            for key, value in topic_model.topic_labels_.items()}
        topics_per_class["Name"] = topics_per_class.Topic.map(topic_names)
        data = topics_per_class.loc[topics_per_class.Topic.isin(selected_topics), :]

        # Add traces
        fig = go.Figure()
        for index, topic in enumerate(selected_topics):
            if index == 0:
                visible = True
            else:
                visible = "legendonly"
            trace_data = data.loc[data.Topic == topic, :]
            topic_name = trace_data.Name.values[0]
            words = trace_data.Words.values
            if normalize_frequency:
                x = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
            else:
                x = trace_data.Frequency
            fig.add_trace(go.Bar(y=trace_data.Class,
                                    x=x,
                                    visible=visible,
                                    marker_color=colors[index % 7],
                                    hoverinfo="text",
                                    name=topic_name,
                                    orientation="h",
                                    hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

        # Styling of the visualization
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        fig.update_layout(
            xaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
            yaxis_title="Class",
            title={
                'text': f"{title}",
                'y': .95,
                'x': 0.40,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=22,
                    color="Black")
            },
            template="simple_white",
            width=width,
            height=height,
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            ),
            legend=dict(
                title="<b>Global Topic Representation",
            )
        )
        return fig


    def topics_per_class_all(self,
                             # classes required if no classes assigned when creating instance
                             classes: List[str] = None,
                             group: List[str] = None,
                             docs: List[str] = None,
                             top_n_topics: int = 10,
                             topics: List[int] = None,
                             custom_labels = None,
                             subplot_titles = ['Topic per class', 'Topic per class'],
                             horizontal_spacing=.05,
                             vertical_spacing=.3,
                             width: int = 1200,
                             height: int = 500) -> go.Figure:
        """
        plot both of freq and relative share
        """
        custom_labels = self._check_flag_cl(custom_labels)
        topic_model = self.topic_model

        docs = self._check_var(docs, self.docs)
        classes = self._check_var(classes, self.classes)
        if (docs is None) or (classes is None):
            print('ERROR!: docs and classes required')
            return None
        else:
            topics_per_class = topic_model.topics_per_class(docs, classes=classes)

        fig = make_subplots(rows=1, cols=2,
                            shared_xaxes=False,
                            shared_yaxes=True,
                            horizontal_spacing=horizontal_spacing,
                            vertical_spacing=0,
                            subplot_titles=subplot_titles)

        # plot 1
        f = self.topics_per_class(topics_per_class,
                                       group=group, docs=docs, classes=classes,
                                       top_n_topics=top_n_topics, topics=topics, custom_labels=custom_labels)

        # update visible to show all topics
        #_ = [fig.add_trace(x.update({'visible':True, 'legendgroup':f'g{i}'}), row=1, col=1) for i, x in enumerate(f.data)]
        _ = [fig.add_trace(x.update({'legendgroup':f'g{i}'}), row=1, col=1) for i, x in enumerate(f.data)]

        # plot 2: relative share of reviews
        f = self.topics_per_class(topics_per_class,
                                       group=group, docs=docs, classes=classes, relative_share=True,
                                       top_n_topics=top_n_topics, topics=topics, custom_labels=custom_labels)

        # update visible to show all topics
        #_ = [fig.add_trace(x.update({'visible':True, 'legendgroup':f'g{i}', 'showlegend':False}), row=1, col=2) for i, x in enumerate(f.data)]
        _ = [fig.add_trace(x.update({'legendgroup':f'g{i}', 'showlegend':False}), row=1, col=2) for i, x in enumerate(f.data)]


        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            width=width,
            height=height,
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Rockwell"
            ),
        )

        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig



class param_study():
    def __init__(self, df_score=None, df_score_stacked=None):
        self.df_score = df_score
        self.df_score_stacked = df_score_stacked


    def import_topic(self, prefix, path_data='.'):
        df_result = read_csv(prefix, path_data)

        # find topic names
        cols = [x for x in df_result.columns if x.isdigit()]

        # convert values to list
        df_result.loc[:, cols] = df_result.loc[:, cols].applymap(lambda x: eval(x) if x is not np.nan else np.nan)

        # convert topics cols to int
        cols_topic = [int(x) for x in cols]
        return df_result.rename(columns=dict(zip(cols, cols_topic)))


    def import_score(self, prefix, path_data='.', header=[0,1]):
        df_score = read_csv(prefix, path_data, header=header)

        # cast topic cols of 'mean', 'median' and 'std' to int
        for x in df_score.columns.get_level_values(0).unique()[1:]:
            cols = df_score[x].columns
            df_score = df_score.rename(columns=dict(zip(cols, cols.astype(int))))

        if self.check_duplicates(df_score) > 0:
            print('WARNING: drop duplicates first.')
        self.df_score = df_score


    def check_duplicates(self, df_score):
        cols_param = list(df_score.columns)[:7]
        a = df_score.loc[:,cols_param].duplicated().sum()
        b = len(df_score)
        print(f'{a} duplicated in {b} param sets')
        return a


    def check_non(self, obj, msg):
        if obj is None:
            print(f'ERROR!: {msg}')
            return True
        else:
            return False


    def stack(self, df_score=None):
        if df_score is None:
            df_score = self.df_score
            if self.check_non(df_score, 'import a score file first'):
                return None

        df_score.columns.names = [None,'Topic']
        cond = (df_score.columns.get_level_values(0) == 'param')
        cols = df_score.columns[cond].tolist()
        self.df_score_stacked = df_score.set_index(cols).rename_axis([x[1] for x in cols]).stack().reset_index()


    def visualize(self, params, kw=['x', 'y', 'color', 'facet_col', 'facet_row'],
                  width=800, height=600, kwa_optional=None, func_plot=px.scatter, marker_size=0, marginal='box',
                  yaxis_range=[0,1]):
        """
        marginal: "histogram", "rug", "box", or "violin". None for no Marginal Distribution Plot
        """
        df_score_stacked = self.df_score_stacked
        if self.check_non(df_score_stacked, 'No stacked score'):
            return None

        kwa = dict(zip(kw, params))
        kwa2 = kwa.copy()
        if marginal is not None:
            if 'facet_col' not in kwa2.keys():
               kwa2['marginal_y'] = marginal
            if 'facet_row' not in kwa2.keys():
               kwa2['marginal_x'] = marginal
        if kwa_optional is not None:
            kwa2.update(kwa_optional)

        # sort params to display param values in ascending order
        # convert numeric color to categorical str for a categorical legend display
        df = df_score_stacked.sort_values(list(kwa.values())[2:])
        if 'color' in kwa.keys():
            df = df.astype({kwa['color']: str})
        fig = func_plot(df, width=width, height=height, **kwa2)

        fig.update_layout(yaxis=dict(range=yaxis_range))

        if marker_size > 0:
            fig.update_traces(marker=dict(size=marker_size),
                            selector=dict(mode='markers'))
        #fig.show()
        return fig


    def _split_param_combinations(self, params, num_plots=5,
                                 kw=['x', 'y', 'color', 'facet_col', 'facet_row'],
                                 kwa_starts = ['Topic', 'mean']):
        """
        split combinations of param sets to num_plots for each plot
        params: list of parameter names
        """
        list_psets_ = [kwa_starts + list(x) for x in list(combinations(params, len(kw)-len(kwa_starts)))]

        divide_list = lambda list_x, n: [list_x[i*n:(i+1)*n] for i in range(len(list_x) // n + max(0, min(1, len(list_x) % n)))]
        list_psets = divide_list(list_psets_, num_plots)

        #print(f'Fig groups from 0 to {len(list_psets)-1} created.')
        return list_psets


    def visualize_all(self, params, num_plots=5,
                      kw=['x', 'y', 'color', 'facet_col', 'facet_row'],
                      kwa_starts = ['Topic', 'mean'],
                      width=1000, height=600, marker_size=3, yaxis_range=[0,1]):
        df_score_stacked = self.df_score_stacked
        if self.check_non(df_score_stacked, 'No stacked score'):
            return None
        list_psets = self._split_param_combinations(params, num_plots=num_plots, kw=kw, kwa_starts=kwa_starts)

        figs = []
        for idx_f, ps in enumerate(list_psets):
            t = lambda i: {'title': f'output_function({idx_f}, [{i}])'}
            fs = [self.visualize(p, kw, width=width, height=height, marker_size=marker_size,
                                 kwa_optional=t(idx_c), yaxis_range=yaxis_range) for idx_c, p in enumerate(ps)]
            figs.append(fs)
        #return figs
        n = len(figs)
        m = sum([len(x) for x in list_psets])
        print(f'{n} figs of {num_plots} param combinations (total {m}) created.')
        print(f'output is a function of {n} figs whose argument is a integer from 0 to {n-1}.')
        return lambda idx_f, idx_c=range(num_plots): [f.show() for i, f in enumerate(figs[idx_f]) if i in idx_c]


class multi_topics_stats():
    def __init__(self, topic_distr, topic_token_distr, df_data=None, df_topic_info=None, col_class=None):
        # A n x m matrix containing the topic distributions for all input documents
        #  with n being the documents and m the topics
        self.topic_distr = topic_distr
        # A list of t x m arrays
        #  with t being the number of tokens for the respective document and m the topics.
        self.topic_token_distr = topic_token_distr
        self.multi_topics_df = None
        self.multi_topics_stats_df = None
        self.df_data = df_data
        self.df_topic_info = df_topic_info # BERTopic.get_topic_info()
        self.sentiment = False
        self.aspect = None
        self.col_class = col_class


    def visualize_distr_per_threshold(self, max_threshhold=0.15, n_topics=5,
                                      vl_max_share = 0.1, vl_decr=0.01,
                                      plot=True, width = 600, height=400, ylabel='share of reviews',
                                      margin_top = 80, margin_bot = 80,
                                      colormap = px.colors.sequential.YlGnBu,
                                      ):
        """
        calculate the distribution of selected topics per review for different threshold levels
         and return optimized threshold to minimize the both documents of lowest number and largest number of topics
        max_threshhold: max of x axis
        n_topics: 1~8
        vl_max_share: see max_share of _search_min_threshold
        """
        topic_distr = self.topic_distr

        tmp_dfs = []
        for thr in tqdm(np.arange(0, max_threshhold, 0.001), leave=False):
            #tmp_df = pd.DataFrame(list(map(lambda x: len(list(filter(lambda y: y >= thr, x))), topic_distr))).rename(
            tmp_df = pd.DataFrame(list(map(lambda x: len(list(filter(lambda y: y > thr, x))), topic_distr))).rename(
                columns = {0: 'num_topics'}
            )
            tmp_df['num_docs'] = 1

            tmp_df['num_topics_group'] = tmp_df['num_topics']\
                .map(lambda x: str(x) if x < n_topics else f'{n_topics}+')

            tmp_df_aggr = tmp_df.groupby('num_topics_group', as_index = False).num_docs.sum()
            tmp_df_aggr['threshold'] = thr

            tmp_dfs.append(tmp_df_aggr)

        print() # line feed after the progress bar of tqdm removed

        num_topics_stats_df = pd.concat(tmp_dfs).pivot(index = 'threshold', values = 'num_docs',
                                  columns = 'num_topics_group').fillna(0)

        num_topics_stats_df = num_topics_stats_df.apply(lambda x: 100.*x/num_topics_stats_df.sum(axis = 1))

        threshold_opt = self._search_min_threshold(num_topics_stats_df, max_share=vl_max_share, decr=vl_decr)
        if threshold_opt is None:
            print(f'WARNING: fail to find threshold with vl_max_share {vl_max_share}.')

        if isinstance(threshold_opt, list):
            threshold_opt = sum(threshold_opt)/len(threshold_opt)

        if plot:
            n_cm = len(colormap)
            n_tps = num_topics_stats_df.columns
            idx = np.random.choice(range(1, n_cm-1), len(n_tps)-2, replace=False)
            idx = [0, *idx, n_cm-1]
            cdm = {k: colormap[i] for k, i in zip(n_tps, np.sort(idx))}

            fig = px.area(num_topics_stats_df,
                title = 'Distribution of number of topics',
                labels = {'num_topics_group': 'number of topics',
                            'value': f'{ylabel}, %'},
                color_discrete_map = cdm,
                width = width, height=height
                )
            fig.update_layout(
                #margin=dict(l=20, r=20, t=100, b=20),
                margin=dict(t=margin_top, b=margin_bot),
                #paper_bgcolor="LightSteelBlue",
                yaxis=dict(range=[0, 100]),
                )


            # plot vline of optimized threshold
            if threshold_opt is not None:
                fig.add_vline(x=threshold_opt, line_width=1, line_dash="dash",
                              #line_color="green",
                              annotation_text=f'threshold {threshold_opt:.3f}',
                              annotation_position="bottom right",
                              annotation_font_color="gray",
                            )


            fig.show()

        #self.num_topics_stats_df = num_topics_stats_df # testing

        return threshold_opt


    def review_docs_wo_topic(self, threshold=0):
        """
        get docs of no topic with which topic_model.visualize_distribution fails to visualize.
         you can see error if run topic_model.visualize_distribution(topic_distr[i])
         where i is in the idx
        """
        topic_distr = self.topic_distr
        func = lambda probas: len([x for x in probas if x > threshold])
        idx = [i for i, x in enumerate(topic_distr) if func(x) == 0]
        n = len(idx) / topic_distr.shape[0]
        print(f'{n*100:.1f} % of docs without any topic (threshold {threshold}).')
        return idx


    def _search_min_threshold(self, num_topics_stats_df, max_share = 0.1, decr=0.01):
        """
        search threshold to minimize the shares of 0 topic and n+ topic
        """
        max_share = max_share * 100
        decr = decr * max_share

        while True:
            cond = (num_topics_stats_df.iloc[:, 0] < max_share)
            cond = cond & (num_topics_stats_df.iloc[:, -1] < max_share)
            thresholds = num_topics_stats_df.loc[cond].index.values
            n = len(thresholds)
            if n > 1:
                prv_th = thresholds
                max_share -= decr
            else:
                if n == 1:
                    threshold = thresholds[0]
                else:
                    threshold = prv_th.mean()
                break

        return threshold


    def _relative_actuality(self, d, sign, sign_percent=1):
        """
        sign: 0 or 1
        """
        if sign == 0:
            return 'no diff'
        if (d >= -sign_percent) and (d <= sign_percent):
            return 'no diff'
        if d < -sign_percent:
            return 'lower'
        if d > sign_percent:
            return 'higher'


    def get_multi_topics_list(self, topic_distr, threshold):
        """
        get the list of multi topics (topics of subsentences) for each document
        """
        multi_topics_list = list(map(
            lambda doc_topic_distr: list(map(
                lambda y: y[0], filter(lambda x: x[1] > threshold,
                                        (enumerate(doc_topic_distr)))
            )), topic_distr
        ))
        return multi_topics_list


    def _get_docs_sentiments(self, docs, multi_topics_list,
                             topic_distr, topic_token_distr,
                             sentiment_func):
        # return list of lists of sentiment labels for topics in multi_topics_list
        # ex) [[positive], [positive, negative], ...]
        # sentiment_func: get a doc and return a list of sentiment labels for topics of subsentence in the doc
        subs_senti = list()
        for i, doc in tqdm(enumerate(docs), desc='Sentiment analysis', total=len(docs)):
            tids = multi_topics_list[i]

            if len(tids) == 0:
                senti = tids
            else:
                #senti = sentiment_func(doc, topic_distr[i], topic_token_distr[i], tids=tids)
                senti = sentiment_func(doc, topic_distr[i], topic_token_distr[i], tids)
            subs_senti.append(senti)
        return subs_senti


    def get_multi_topics_df(self, threshold=0, df_data=None, cols_add=None,
                            sentiment=None, sentiment_func=None, docs=None):
        """
        get df of topic, docu id and class.
         count of id is num of all subsentences, which is greater than num of docs(nunique of id).
        df_data: dataframe of docs including document id and cols_add (such as document class)
        cols_add: list of columns of df_data for multi_topics_df
        sentiment: None, True, False. None to use self.sentiment
        sentiment_func: get doc, topic_distr and topic_token_distr of the doc, topics for subsentences of the doc as input
        """
        df_data = self._check_var(df_data, self.df_data)
        if df_data is None:
            print('ERROR: No df_data assigned')
            return None

        topic_distr = self.topic_distr

        # define topics with probability > threshold for each document
        # This approach will help us to reduce the number of outliers
        multi_topics_list = self.get_multi_topics_list(topic_distr, threshold)
        df_data['multiple_topics'] = multi_topics_list

        sentiment = self._check_var(sentiment, self.sentiment)
        if sentiment:
            if (sentiment_func is None) or (docs is None):
                print('WARNING!: working without sentiment as missing inputs (sentiment_func or docs).')
                sentiment = False
            else:
                df_data['sentiment'] = self._get_docs_sentiments(docs, multi_topics_list,
                                                                 topic_distr, self.topic_token_distr,
                                                                 sentiment_func)
                self.sentiment = sentiment

        # create multi_topics_df which shows many documents have multiple topics
        tmp_data = []
        for rec in df_data.to_dict('records'):
            if len(rec['multiple_topics']) != 0:
                multi_topics = rec['multiple_topics']
                if sentiment:
                    topic_sentiments = rec['sentiment']
            else:
                multi_topics = [-1] # assign outlier topic
                if sentiment:
                    topic_sentiments = [None]

            for i, topic in enumerate(multi_topics):
                kw = {
                    'topic': topic, # topic id
                    'id': rec['id'], # doc id
                }
                # update with additional keys such as class and doc
                if cols_add is not None:
                    kw.update({x: rec[x] for x in cols_add})
                if sentiment:
                    kw.update({'sentiment': topic_sentiments[i]})
                tmp_data.append(kw)

        self.multi_topics_df = pd.DataFrame(tmp_data)
        return self.multi_topics_df


    def get_multi_topics_info(self, df_topic_info=None, aspect=None):
        """
        Get information about each topic including its ID, frequency, share, and name from multi_topics_df.
        share is based on num of subsentences, not on num of docs
        """
        df_topic_info = self._check_var(df_topic_info, self.df_topic_info)
        if df_topic_info is None:
            print('ERROR: No df_topic_info assigned')
            return None

        multi_topics_df = self.multi_topics_df
        if multi_topics_df is None:
            print('ERROR: Run get_multi_topics_df first')
            return None

        aspect = self._check_aspect(df_topic_info, aspect)
        if aspect is None:
            return None

        top_multi_topics_df = multi_topics_df.groupby('topic', as_index = False).id.nunique()
        # share is ratio of topic to all subsentences (not to num of documents)
        top_multi_topics_df['share_to_docs'] = 100.*top_multi_topics_df.id/top_multi_topics_df.id.sum()
        top_multi_topics_df['topic_repr'] = top_multi_topics_df.topic.map(
            lambda x: self._get_topic_representation(x, df_topic_info, aspect=aspect)
        )

        return top_multi_topics_df.sort_values('id', ascending = False).rename(columns={'id': 'freq'})


    def create_multi_topics_stats(self, col_class=None, alpha=0.05, warning_ztest_r=0.2,
                                  # inputs for get_multi_topics_df
                                  threshold=0, df_data=None, cols_add=None,
                                  sentiment=None, sentiment_func=None, docs=None):
        """
        create stats from self.multi_topics_df
        """
        # check col_class which is requisite
        col_class = self._check_var(col_class, self.col_class)
        if col_class is None:
            print('ERROR: Set col_class.')
            return None

        # check and set sentiment
        sentiment = self._check_var(sentiment, self.sentiment)
        if sentiment is not None:
            # update for visualize_class_by_topic
            self.sentiment = sentiment

        # get multi_topics_df
        multi_topics_df = self.multi_topics_df
        if multi_topics_df is None:
            if sentiment: # Run get_multi_topics_df with sentiment
                kwa = dict(sentiment=sentiment, sentiment_func=sentiment_func, docs=docs)
            else:
                kwa = dict(sentiment=sentiment)

            multi_topics_df = self.get_multi_topics_df(threshold=threshold, df_data=df_data, cols_add=[col_class], **kwa)
        else:
            if col_class not in multi_topics_df.columns:
                if sentiment: # Run get_multi_topics_df with sentiment
                    kwa = dict(sentiment=sentiment, sentiment_func=sentiment_func, docs=docs)
                else:
                    kwa = dict(sentiment=sentiment)

                multi_topics_df = self.get_multi_topics_df(threshold=threshold, df_data=df_data, cols_add=[col_class], **kwa)
            else:
                if sentiment and ('sentiment' not in multi_topics_df.columns):
                    # Run get_multi_topics_df with sentiment
                    kwa = dict(sentiment=sentiment, sentiment_func=sentiment_func, docs=docs)
                    multi_topics_df = self.get_multi_topics_df(threshold=threshold, df_data=df_data, cols_add=[col_class], **kwa)

        # create multi_topics_stats_df
        tmp_data = []
        for cls in multi_topics_df[col_class].unique():
            for topic in multi_topics_df.topic.unique():
                cond_class = (multi_topics_df[col_class] == cls)
                cond_topic = (multi_topics_df.topic == topic)
                if sentiment:
                    for senti in multi_topics_df.loc[cond_class & cond_topic].sentiment.unique():
                        cond_senti = (multi_topics_df.sentiment == senti)
                        tmp_data.append({
                            col_class: cls,
                            'topic_id': topic,
                            'sentiment': senti,
                            f'total_{col_class}_docs': multi_topics_df[cond_class].id.nunique(),
                            f'topic_{col_class}_docs': multi_topics_df[cond_class & cond_topic & cond_senti].id.nunique(),
                            f'other_{col_class}s_docs': multi_topics_df[~cond_class].id.nunique(),
                            f'topic_other_{col_class}s_docs': multi_topics_df[~cond_class & cond_topic & cond_senti].id.nunique()
                        })
                else: # multi_topics_df will not have the sentiment column
                    tmp_data.append({
                        col_class: cls,
                        'topic_id': topic,
                        f'total_{col_class}_docs': multi_topics_df[cond_class].id.nunique(),
                        f'topic_{col_class}_docs': multi_topics_df[cond_class & cond_topic].id.nunique(),
                        f'other_{col_class}s_docs': multi_topics_df[~cond_class].id.nunique(),
                        f'topic_other_{col_class}s_docs': multi_topics_df[~cond_class & cond_topic].id.nunique()
                    })

        multi_topics_stats_df = pd.DataFrame(tmp_data)
        multi_topics_stats_df[f'topic_{col_class}_share'] = 100*multi_topics_stats_df[f'topic_{col_class}_docs']/multi_topics_stats_df[f'total_{col_class}_docs']
        multi_topics_stats_df[f'topic_other_{col_class}s_share'] = 100*multi_topics_stats_df[f'topic_other_{col_class}s_docs']/multi_topics_stats_df[f'other_{col_class}s_docs']

        # testing
        #return (multi_topics_df, multi_topics_stats_df, col_class)

        # define class difference by practical significance (alpha)
        def calc_paval(x1, x2, n1, n2):
            # implemented as i cannot catch exception in proportions_ztest
            if x1 + x2 == 0:
                return None
            else:
                return proportions_ztest(count = [x1, x2],
                                         nobs = [n1, n2],
                                         alternative = 'two-sided'
                                        )[1]

        multi_topics_stats_df['difference_pval'] = list(map(
            calc_paval,
            multi_topics_stats_df[f'topic_other_{col_class}s_docs'],
            multi_topics_stats_df[f'topic_{col_class}_docs'],
            multi_topics_stats_df[f'other_{col_class}s_docs'],
            multi_topics_stats_df[f'total_{col_class}_docs']
        ))

        df = multi_topics_stats_df['difference_pval']
        x = df.isna().sum() / len(df)
        if x > warning_ztest_r:
            print(f'WARNING!: topics to no {col_class} more than {x*100} %')

        multi_topics_stats_df['sign_difference'] = multi_topics_stats_df['difference_pval'].map(
            lambda x: 1 if x <= alpha else 0
        )

        # diff_significance_total for color distinction in visualize_class_by_topic
        multi_topics_stats_df['diff_significance_total'] = list(map(
            self._relative_actuality,
            multi_topics_stats_df[f'topic_{col_class}_share'] - multi_topics_stats_df[f'topic_other_{col_class}s_share'],
            multi_topics_stats_df['sign_difference']
        ))

        if sentiment: # create text label for sentiment share of each class
            df_tmp = multi_topics_stats_df.groupby([col_class, 'topic_id'])[f'topic_{col_class}_share'].sum().rename('sentiment_share').reset_index()
            multi_topics_stats_df = pd.merge(multi_topics_stats_df, df_tmp, on=[col_class, 'topic_id'], how='left')
            multi_topics_stats_df['sentiment_share'] = (multi_topics_stats_df[f'topic_{col_class}_share']
                                                        .div(multi_topics_stats_df['sentiment_share'])
                                                        .apply(lambda x: f'{x:.0%}'))
                                      
        print('stats for visualize_class_by_topic created.')

        #return multi_topics_stats_df
        self.multi_topics_stats_df = multi_topics_stats_df


    def _check_var(self, var_arg, var_self):
        if var_arg is None:
            var_arg = var_self
        return var_arg


    def _check_aspect(self, df_topic_info, aspect=None, aspect_default='Representation'):
        aspect = self._check_var(aspect, self.aspect)
        return check_topic_aspect(df_topic_info, aspect, aspect_default=aspect_default, start_idx=3)


    def _get_topic_representation(self, topic, df_topic_info, aspect=None):
        data = df_topic_info.loc[df_topic_info.Topic==topic].iloc[0][aspect]
        self.aspect = aspect # save for next use

        #return ', '.join(data[:5]) + ', <br>         ' + ', '.join(data[5:])
        return ', '.join(data)


    def get_color_significance(self, rel):
        if rel == 'no diff':
            return color_qual.Set2[7] # grey
        if rel == 'lower':
            return color_qual.Set2[1] # orange
        if rel == 'higher':
            return color_qual.Set2[0] # green


    def visualize_class_by_topic(self, topic, df_topic_info=None,
                                 width=700, height=60, ylabel='share of reviews',
                                 title_length=60, barmode='stack',
                                 sentiment_order = SENTIMENT_LABELS,
                                 sentiment_color = [color_qual.Plotly[x] for x in [0,7,1]],
                                 class_order_ascending = None,
                                 aspect=None,
                                 title_font_size=14,
                                 class_label_length=0,
                                 horizontal_bar=True,
                                 margin_width=100, margin_height=100,
                                 filename=None, noshow=False):
        """
        barmode: 'stack', 'group', 'overlay', 'relative'
        class_order_ascending: None, True, False
        filename: file name with path. create file in working dir if no path exits
        noshow: set to True when just saving fig
        """

        df_topic_info = self._check_var(df_topic_info, self.df_topic_info)
        if df_topic_info is None:
            print('ERROR: No df_topic_info assigned')
            return None

        multi_topics_stats_df = self.multi_topics_stats_df
        if multi_topics_stats_df is None:
            print('ERROR: Run create_multi_topics_stats first')
            return None

        aspect = self._check_aspect(df_topic_info, aspect)
        if aspect is None:
            return None

        col_class = self.col_class

        cond = (multi_topics_stats_df.topic_id == topic)
        cond = cond & (multi_topics_stats_df[f'topic_{col_class}_share'] > 0)
        topic_stats_df = multi_topics_stats_df.loc[cond]\
            .sort_values(f'total_{col_class}_docs', ascending = False).set_index(col_class)

        # set plot title
        title = self._get_topic_representation(topic_stats_df.topic_id.min(), df_topic_info, aspect)
        indent=' '*2
        title = split_str(title, length=title_length, split='<br>', indent=indent)
        #title = f'Topic_{topic}:<br>{indent}{title}'
        title = f'Topic_{topic}: {title}'

        # set bar colors
        sentiment_color = dict(zip(sentiment_order, sentiment_color))
        category_orders = {'sentiment': sentiment_order}

        # set plot options depending on sentiment
        if self.sentiment:
            showlegend = True
            color = 'sentiment' # select legend
            kw_up_traces = {'marker_line_width': 1.5, 'opacity': 0.9}
            bar_label = 'sentiment_share'
        else:
            showlegend = False
            color = None
            diff_colors = list(map(self.get_color_significance,
                                  topic_stats_df.diff_significance_total))
            kw_up_traces = {'marker_line_width': 1.5, 'opacity': 0.9,
                            'marker_color': diff_colors, 'marker_line_color': diff_colors}
            bar_label = f'topic_{col_class}_share'


        df_plot = topic_stats_df.reset_index()
        # split wine name for plot labels
        if class_label_length > 0:
            df_plot[col_class] = df_plot[col_class].apply(lambda x: split_str(x, length=class_label_length, split='<br>'))

        if horizontal_bar:
            x = f'topic_{col_class}_share'
            y = col_class
            orientation='h'
            plot_total_share = lambda fig, x, **kw: fig.add_vline(x=x, **kw)
            kw_uplout = {'yaxis_title': None}
            height = max(df_plot[y].nunique() * height + margin_height, 300)
            height = min(height, 800)
        else:
            x = col_class
            y = f'topic_{col_class}_share'
            orientation='v'
            plot_total_share = lambda fig, y, **kw: fig.add_hline(y=y, **kw)
            kw_uplout = {'xaxis_title': None}
            width = min(df_plot[y].nunique() * width + margin_width, 1200)

        # testing
        #return (df_plot, x, y, orientation, title, color, category_orders,
        #        sentiment_color, barmode, col_class, ylabel, width, height,
        #        px, showlegend, title_font_size, kw_uplout, kw_up_traces)

        fig = px.bar(df_plot,
                     x = x,
                     y = y,
                     orientation=orientation,
                     title = title,
                     color = color,
                     category_orders = category_orders,
                     color_discrete_map = sentiment_color,
                     barmode=barmode,
                     #text_auto = '.0f %',
                     text = bar_label,
                     labels = {f'topic_{col_class}_share': f'{ylabel}, %'},
                     hover_data=['diff_significance_total'],
                     width=width, height=height)

        fig.update_layout(showlegend=showlegend,
                          title_font_size=title_font_size,
                          xaxis={'automargin': True}, # wine name still overlapping
                          **kw_uplout
                          )
        fig.update_traces(**kw_up_traces)

        #return (topic_stats_df, df_plot, col_class) # testing

        # calc total share for horizontal line plot
        # topic_total_share is the percent of num of documents of a topic to num of all documents
        # it's different with share from get_multi_topics_info which is based on num of subsentences.
        df = topic_stats_df.loc[topic_stats_df.index[0]] # pick one class as any class retruns same result in the end
        topic_total_share = 100.*((df[f'topic_{col_class}_docs'] + df[f'topic_other_{col_class}s_docs'])\
                                   /(df[f'total_{col_class}_docs'] + df[f'other_{col_class}s_docs']))
        if self.sentiment:
            topic_total_share = topic_total_share.sum()

        plot_total_share(fig, topic_total_share,
                         line_dash="dot",
                         line_width=1.2,
                         line_color='gray',
                         #annotation_text=f'Topic_{topic} share {topic_total_share:.0f}%',
                         annotation_text=f'topic share {topic_total_share:.0f}%',
                         annotation_position="bottom right",
                         #annotation_font_size=20,
                         annotation_font_color="gray",
                        )

        if class_order_ascending is not None:
            if class_order_ascending:
                fig.update_yaxes(categoryorder='category ascending')
            else:
                fig.update_yaxes(categoryorder='category descending')
                                     
        if not noshow:
            fig.show()

        # save fig as json
        if filename is not None:
            path = '/'.join(filename.split('/')[:-1])
            if not os.path.isdir(path):
                # to save on working dir
                filename = filename.split('/')[-1]
            f = f'{filename}_topic{topic:02}.json'
            pio.write_json(fig, f)

        return topic_stats_df


    def get_multi_topic_docs(self, topic, class_name=None, num_print=5, length_print=120,
                             docs=None, classes_all=None, aspect=None,
                             df_topic_info=None, col_class=None):
        """
        get docs of the multi topic and print the num_print of docs.
        docs: set as kwa following the convention of utils.get_topic_docs
        """
        if docs is None:
            print('ERROR: No docs assigned.')
            return None

        multi_topics_df = self.multi_topics_df
        if multi_topics_df is None:
            print('ERROR: Run get_multi_topics_df first')
            return None

        cond = (multi_topics_df.topic == topic)
        if class_name is not None:
            col_class = self._check_var(col_class, self.col_class)
            if col_class is None:
                print('ERROR: Set col_class.')
                return None
            cond = cond & multi_topics_df[col_class].str.lower().str.contains(class_name.lower())

        topic_docs = [docs[x] for x in multi_topics_df.loc[cond].id.unique()]

        # get topic label
        df_topic_info = self._check_var(df_topic_info, self.df_topic_info)
        if df_topic_info is None:
            print('ERROR: No df_topic_info assigned')
            return None
        aspect = self._check_aspect(df_topic_info, aspect)
        if aspect is None:
            return None
        topic_name = self._get_topic_representation(topic, df_topic_info, aspect)

        # print docs of num_print
        if num_print > 0:
            if len(topic_docs) > num_print:
                print(f'Printing {num_print} docs from {len(topic_docs)}')
                docs_print = np.random.choice(topic_docs, num_print, replace=False)
            else:
                docs_print = topic_docs

            print(f'Topic {topic}: {topic_name}')
            _ = [print_with_line_feed(f'-. {x}', length_print, indent='') for x in docs_print]

        return topic_docs


class multi_topics_sentiment():
    def __init__(self, topic_model, tokenizer=None, sentiment_analysis=None, max_sequence_length=2000):
        self.topic_labels = {topic: label for topic, label in topic_model.topic_labels_.items() if topic > -1}
        self.sentiment_analysis = sentiment_analysis
        self.max_sequence_length = max_sequence_length

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = topic_model.vectorizer_model.build_tokenizer()

        self.tm_visualize_distribution = topic_model.visualize_distribution

    def _check_var(self, var_arg, var_self):
        if var_arg is None:
            var_arg = var_self
        return var_arg

    def visualize_distribution(self,
                               topic_distr_doc,
                               min_probability=0.015,
                               plot=True,
                               pattern = r'<b>\s*Topic\s*(\d+)</b>',
                               **kwargs):
        """
        get the topics of a document from the distribution of topic probabilities
        """
        probs = self.tm_visualize_distribution(topic_distr_doc, min_probability=min_probability, **kwargs)
        if plot:
            probs.show()

        tid_list = []
        for s in probs.data[0]['y']:
            # Search for the pattern
            match = re.search(pattern, s, re.IGNORECASE)
            tid_list.append(int(match.group(1)))

        return tid_list


    def get_token_distribution(self, doc, topic_token_distr_doc, tokenizer, normalize=False,
                               index_label=False):
        """
        tokenizer: CountVectorizer.build_tokenizer(). check BERTopic.vectorizer_model for the vectorizer
        topic_token_distr_doc: the doc's topic_token_distribution
        """
        # Tokenize document
        tokens = tokenizer(doc)

        if len(tokens) == 0:
            raise ValueError("Make sure that your document contains at least 1 token.")

        # Prepare dataframe with results
        if normalize:
            df = pd.DataFrame(topic_token_distr_doc / topic_token_distr_doc.sum()).T
        else:
            df = pd.DataFrame(topic_token_distr_doc).T

        df.columns = [f"{token}_{i}" for i, token in enumerate(tokens)]
        df.columns = [f"{token}{' '*i}" for i, token in enumerate(tokens)]
        if index_label: # index is topic label
            df.index = list(self.topic_labels.values())
        else: # index is topic id
            df.index = list(self.topic_labels.keys())
        # drop topics of no distribution for any token in the doc
        df = df.loc[(df.sum(axis=1) != 0), :]

        # dataframe indicating the best fitting topics for each token
        return df


    def get_subsentences(self, df_token_distr, tids=None, min_token_proba=0, n_padding=0):
        """
        extract subsentences (sets of tokens) related to each topic from a document
        return dict of topic (id with set of words) to list of subsentences
        df_token_distr: token-level distributions of a document. index must be topic id
        n_padding: number of extra token to add to a subsentence to give a context for sentiment analysis
        """
        if tids is None:
            dict_token_distr = df_token_distr.to_dict('index')
        else:
            dict_token_distr = {k: v for k, v in df_token_distr.to_dict('index').items() if k in tids}

        if len(dict_token_distr) == 0:
            print('ERROR: no topic to get its subsentences')
            return None

        subsentences = {} # topic to subsentences
        for topic, dist_dict in dict_token_distr.items():
            blocks = [] # subsentences for each topic
            block = [] # temp list for a subsentence building

            token_all = list(dist_dict.keys())
            for i, (token, dist) in enumerate(dist_dict.items()):
                if dist > min_token_proba:
                    if (i * n_padding > 0) and (len(block) == 0):
                        start = max(i-n_padding, 0)
                        block = [x.strip() for x in token_all[start:i]]
                    block.append(token.strip())
                else:
                    if len(block) > 0:
                        if n_padding > 0:
                            block.extend([x.strip() for x in token_all[i:i+n_padding]])
                        blocks.append(' '.join(block))
                    block = []

            subsentences[topic] = blocks
        #return subsentences
        return {self.topic_labels[k]: v for k, v in subsentences.items()}


    def topic_sentiment(self, topic_subsentences, sentiment_analysis,
                        label_only=False, min_score=0,
                        max_sequence_length=2000):
        senti = {}
        for topic, subs in topic_subsentences.items():
            res = sentiment_analysis([x[:max_sequence_length] for x in subs],
                                     # return max score only
                                     return_all_scores=False)
            scores = []
            for x in res:
                label = x["label"]
                score = x["score"]
                if score <= min_score:
                    label = 'n/a'

                if label_only:
                    s = label
                else:
                    s = f'({label}) {score:.3f}'
                scores.append(s)
            senti[topic] = scores

        return senti


    def get_topic_score(self, tid, doc_subs, doc_senti):
        try:
            t_list = list(doc_subs.keys())
            tp = t_list[tid]
            return dict(zip(doc_subs[tp], doc_senti[tp]))
        except Exception as e:
            print(f'ERROR: {e}')
            return


    def get_docu_sentiments(self, doc, topic_distr_doc, topic_token_distr_doc,
                            tokenizer=None, sentiment_analysis=None,
                            tids=None, min_proba = 0.015, pattern = r'<b>\s*Topic\s*(\d+)</b>',
                            min_token_proba = 0,
                            single_subsentence=True,
                            n_padding = 0,
                            #print_result=False,
                            topic_id = True,
                            label_only=False,
                            min_score=50,
                            max_sequence_length=None
                            ):
        """
        return two dict, 1st one is topic to subsentences,
         the other is topic to sentiments of the subsentences
        """
        tokenizer = self._check_var(tokenizer, self.tokenizer)
        if tokenizer is None:
            print('ERROR: No tokenizer assigned.')
            return None

        sentiment_analysis = self._check_var(sentiment_analysis, self.sentiment_analysis)
        max_sequence_length = self._check_var(max_sequence_length, self.max_sequence_length)
        if sentiment_analysis is None:
            print('ERROR: No sentiment_analysis assigned.')
            return None

        if tids is None:
            tids = self.visualize_distribution(topic_distr_doc,
                                               min_probability=min_proba,
                                               pattern=pattern,
                                               plot=False)

        df_token_distr = self.get_token_distribution(doc, topic_token_distr_doc, tokenizer)
        subs = self.get_subsentences(df_token_distr, tids=tids, min_token_proba=min_token_proba, n_padding=n_padding)

        # concat all subsentences for each topic
        if single_subsentence:
            subs = {k: ['. '.join(v)] for k,v in subs.items()}

        # ex) senti = {0 : [positive], 5: [negative]} if single_subsentence and label_only are True
        senti = self.topic_sentiment(subs, sentiment_analysis,
                                     label_only=label_only, min_score=min_score,
                                     max_sequence_length=max_sequence_length)

        if topic_id:
            subs = {tids[i]: v for i, v in enumerate(subs.values())}
            senti = {tids[i]: v for i, v in enumerate(senti.values())}

        return (subs, senti)



    def mts_sentiment_func(self, n_padding=1, min_score=0):

        def sentiment_func(doc, topic_distr_doc, topic_token_distr_doc, tids):
            _, senti = self.get_docu_sentiments(doc, topic_distr_doc, topic_token_distr_doc, tids=tids,
                                                # required settings
                                                min_proba = 0, min_token_proba = 0,
                                                single_subsentence=True, label_only=True,
                                                # optional
                                                n_padding=n_padding, min_score=min_score
                                                )
            return [x for x_list in senti.values() for x in x_list]

        return sentiment_func
