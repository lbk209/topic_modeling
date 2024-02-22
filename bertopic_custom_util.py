import collections
import os
import pandas as pd
import numpy as np

from typing import List, Union

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from itertools import combinations

def read_csv(file, path_data, **kwargs):
    """
    kwargs: keyword args for pd.read_csv
    """
    files = [x for x in os.listdir(path_data) if x.startswith(file)]
    
    if len(files) == 0:
        print('ERROR!: No csv to read')

    df_reviews = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f'{path_data}/{f}', **kwargs)
        df_reviews = pd.concat([df_reviews, df])

    return df_reviews.reset_index(drop=True)


def print_with_line_feed(input_string, line_length=50):
    words = input_string.split()
    current_line_length = 0

    for word in words:
        if current_line_length + len(word) <= line_length:
            print(word, end=" ")
            current_line_length += len(word) + 1  # +1 for the space
        else:
            print()  # Start a new line
            print(f'  {word}', end=" ")
            current_line_length = len(word) + 1

    print()  # Ensure the last line is printed



class utils():
    def __init__(self, topic_model, reduced_embeddings=None):
        self.topic_model = topic_model
        # for visualize_documents
        self.reduced_embeddings = reduced_embeddings
        self.count_visualize = 0
        
    def print_topic_info(self):
        """
        print number of topics and percentage of outliers
        """
        df = self.topic_model.get_topic_info()

        a = len(df) - 1
        print(f'num of topics: {a}')

        a = df.loc[df.Topic == -1]['Count']
        if a.count() > 0:
            a = a.values[0]/df['Count'].sum()
        else:
            a = 0
        print(f'outliers: {a:.3f}')
        
        return df


    def print_custom_labels(self, list_tid=None, min_count=0, length=120):
        """
        list_tid: topics to print custom labels
        length: number of chars to print every line
        dict_label: dict of topic and custom label
        """
        tid_all = self.topic_model.topic_labels_.keys()
        dict_label = dict(zip(tid_all, self.topic_model.custom_labels_))
        
        if list_tid is None:
            df = self.topic_model.get_topic_info()
            list_tid = df.loc[(df.Topic>-1) & (df.Count>=min_count)].Topic.to_list()
        else:        
            if not isinstance(list_tid, list):
                list_tid = [list_tid]
        
        dict_label = {k:v for k,v in dict_label.items() if k in list_tid}
        _ = [print_with_line_feed(v, length) for k,v in dict_label.items()]

        return dict_label


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
        
        if self.count_visualize > 0:
            print('WARNING: create the instance of visualize again.')
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
        else:
            list_labels = topic_model.topic_labels_.values()

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
        
        
    def visualize(self, docs, classes=None):
        self.count_visualize += 1
        # return a instance of the class visualize
        return visualize(self.topic_model, docs, classes, self.reduced_embeddings)


class visualize():
    def __init__(self, topic_model, docs, classes=None, reduced_embeddings=None, custom_labels=False):
        self.topic_model = topic_model
        self.docs = docs
        self.classes = classes
        self.reduced_embeddings = reduced_embeddings
        self.custom_labels = custom_labels
        
    def _check_flag_cl(self, custom_labels):
        if custom_labels is None:
            custom_labels = self.custom_labels
        return custom_labels
        
    def visualize_documents(self, list_tid=None, custom_labels=None,
                            hide_annotations=True, hide_document_hover=False, **kwargs):
        """
        custom_labels: set to False if custom label is long enough to fill the whole fig
        """
        custom_labels = self._check_flag_cl(custom_labels)
        titles = [x[:100] for x in self.docs]
        if list_tid is None:
            list_tid = range(20)

        return self.topic_model.visualize_documents(titles, topics=list_tid, custom_labels=custom_labels,
                                        hide_annotations=hide_annotations, hide_document_hover=hide_document_hover,
                                        reduced_embeddings=self.reduced_embeddings, **kwargs)
                                        

    def topics_per_param(self, df_result, res_docs,
                                   ncols=4, top_n_topics=None,
                                   horizontal_spacing=.05,
                                   vertical_spacing=.3,
                                   width = 350, height = 350
                                   ):
        """
        grid plot of classes (class is param)
        """
        subplot_titles = [x for x in df_result.columns if not isinstance(x, int)]
        nrows = len(subplot_titles)//ncols+1

        fig = make_subplots(rows=nrows, cols=ncols,
                            shared_xaxes=False,
                            horizontal_spacing=horizontal_spacing,
                            vertical_spacing=vertical_spacing / nrows if nrows > 1 else 0,
                            subplot_titles=subplot_titles)

        row, col = 1, 1
        for i, _ in enumerate(subplot_titles):
            classes = df_result.iloc[:, i].apply(str)
            topics_per_class = self.topic_model.topics_per_class(res_docs, classes=classes)

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
                                       docs: List[str],
                                       classes: List[str],
                                       top_n_topics: int = 10,
                                       topics: List[int] = None,
                                       group: List[str] = None,
                                       custom_labels = None,
                                       horizontal_spacing=.05,
                                       vertical_spacing=.3,
                                       width: int = 1200,
                                       height: int = 500) -> go.Figure:
        custom_labels = self._check_flag_cl(custom_labels)                        
        topic_model = self.topic_model
        
        subplot_titles = ['Topic per class', 'Topic per class']

        fig = make_subplots(rows=1, cols=2,
                            shared_xaxes=False,
                            shared_yaxes=True,
                            horizontal_spacing=horizontal_spacing,
                            vertical_spacing=0,
                            subplot_titles=subplot_titles)

        topics_per_class = topic_model.topics_per_class(docs, classes=classes)

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
        self.split_paramsets = None
        self.split_kw = None


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
                     width=800, height=600, kwa_optional=None, func_plot=px.scatter, marker_size=0, marginal='box'):
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
        fig = func_plot(df_score_stacked.sort_values(list(kwa.values())[2:]).astype({kwa['color']: str}),
                        width=width, height=height, **kwa2)
        
        fig.update_layout(yaxis=dict(range=[0,1]))
        
        if marker_size > 0:
            fig.update_traces(marker=dict(size=marker_size),
                            selector=dict(mode='markers'))
        #fig.show()
        return fig

    
    def split_param_combinations(self, params, num_plots=5, 
                                 kw=['x', 'y', 'color', 'facet_col', 'facet_row'],
                                 kwa_starts = ['Topic', 'mean']):
        """
        split combinations of param sets to num_plots for each plot
        params: list of parameter names
        """
        list_psets_ = [kwa_starts + list(x) for x in list(combinations(params, len(kw)-len(kwa_starts)))]

        divide_list = lambda list_x, n: [list_x[i*n:(i+1)*n] for i in range(len(list_x) // n + max(0, min(1, len(list_x) % n)))]
        list_psets = divide_list(list_psets_, num_plots)

        print(f'Plot groups from 0 to {len(list_psets)-1} created.')
        self.split_paramsets = list_psets
        self.split_kw = kw


    def visualize_all(self, nth, width=1000, height=600, marker_size=3):
        df_score_stacked = self.df_score_stacked
        list_psets = self.split_paramsets
        if (self.check_non(df_score_stacked, 'No stacked score') or
            self.check_non(list_psets, 'Split combinations first')):
            return None

        kw = self.split_kw
        figs = []
        for p in list_psets[nth]:
            f = self.visualize(p, kw, width=width, height=height, marker_size=marker_size)
            figs.append(f)
        return figs
