import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='folder path', type=str, default='')
parser.add_argument('-l', '--log_file', help='log file path', type=str, default='')
parser.add_argument('-c', '--clusters', help='most representative clusters ids', nargs='*', type=int, default=[])

def nodes_dict (edges, groups):
    '''
    Function to create dictionary with nodes attributes
    
    Argguments:
        edges{DataFrame} -- dataset with the edges in the graph
        groups{DataFrame} -- dataset with the modularity class of each node in the graph
    
    Returns:
        [dict] -- nodes attributes
    '''
    nodes = np.unique([edges['source'],edges['target']])
    d = {node:{
            'gp':None,
            'type':None,
            'edges':{
                    'source':{'id':[], 'gp':[]},
                    'target':{'id':[], 'gp':[]}}
            }
    for node in nodes}

     
    for key in d.keys():
        # node's modularity
        d[key]['gp'] = groups[groups['name'] == key]['group']
        # source nodes --> nodes with node as target
        source = list(edges[edges['target'] == key]['source'])
        gp_source = list(groups[groups['name'].isin(source)]['group'])
            
        d[key]['edges']['source']['id'] += source
        d[key]['edges']['source']['gp'] += gp_source
        # target nodes --> nodes with node as source
        target = list(edges[edges['source'] == key]['target'])
        gp_target = list(groups[groups['name'].isin(target)]['group'])
        
        d[key]['edges']['target']['id'] += target
        d[key]['edges']['source']['gp'] += gp_target
        return d

def filter_graph (edges, groups, cluster_ids):
    '''
    Function to filter the graph, keeping only the most relevant modularity classes
    
    Argguments:
        edges{DataFrame} -- dataset with the edges in the graph
        groups{DataFrame} -- dataset with the modularity class of each node in the graph
        cluster_ids{list} -- optional list with the relevant modularity class ids
    
    Returns:
        [list] -- list of remaining edges and groups
    '''
    
    if len(cluster_ids) == 0: 

        # Removes clusters with less than 5% of nodes
        freqs = groups['group'].value_counts()
        ids = freqs[freqs > .05*np.sum(freqs)].index
        groups = groups[groups['group'].isin(ids)] 
    else:
        groups = groups[groups['group'].isin(cluster_ids)]

    edges = edges[(edges['source'].isin(groups['name'])) & (edges['target'].isin(groups['name']))]
    
    # Join edges and nodes' modularity
    edges = edges.merge(groups, left_on = 'source', right_on = 'name')
    edges = edges.merge(groups, left_on = 'target', right_on = 'name') 
    edges = edges.drop(['name_x','name_y'], 1)
    edges.columns = ['source','target','gp_source','gp_target']

    return [edges, groups]

def get_internal (edges):
    '''
    Function to create set of internal nodes
    
    Argumrnts:
        edges{DataFrame} -- dataset with the edges in the graph
    
    Returns:
        [list] -- list of internal nodes
    '''

    nodes = np.unique([edges['source'],edges['target']])
    I = []

    for node in nodes:
        # edges linking this node to others
        aux = edges[(edges['source'] == node) | (edges['target'] == node)]
        # modularity of nodes linked with this node
        gp = np.unique([aux['gp_source'].values,aux['gp_target'].values])
        # keeping only if the node is linked with nodes on the same cluster (modularity)
        if len(gp) == 1:
            I.append(node)
            
    return I
 
def get_boundaries (edges, groups, I):
    '''
    Function to create set of nodes on boundaries
    
    Arguments:
        edges{DataFrame} -- dataset with the edges in the graph
        groups{DataFrame} -- dataset with the modularity class of each node in the graph
        I{list} -- list with internal nodes
    
    Returns:
        [list] -- list of nodes on boundaries
    '''

    # condition 1 in the article
    c1 = edges[edges['gp_source'] != edges['gp_target']]
    nodes = np.unique([c1['source'], c1['target']])  
    B= []
    
    for node in nodes:
        # node modulatiry class id
        gp = groups[groups['name'] == node]['group']
        # internal nodes of this modularity
        I_gp = groups[(groups['name'].isin(I)) & (groups['group'].isin(gp))]
        
        # internal nodes linked with the node (condition 2)
        I_edges = edges[(edges['source'] == node) | (edges['target'] == node)]
        I_edges = I_edges[I_edges['gp_source'] == I_edges['gp_target']]
        
        if I_edges.shape[0] > 0:
            users = np.unique([I_edges['source'].values,I_edges['target'].values])
            if any(u in I_gp['name'].values for u in users):
                B.append(node) 
                
    return B

def get_internal_edges (edges, I, B):
    '''
    Function to create set of edges linking internal nodes with nodes on boundaries
    
    Arguments:
        edges{DataFrame} -- dataset with the edges in the graph
        I{list} -- list with internal nodes
        B{list} -- list with nodes on boundaries
    Returns:
        [DataFrame] -- edges of internal and boundaries nodes
    '''
    # edges where source is boundary and target is internal
    e_i1 = edges[(edges['source'].isin(B)) & (edges['target'].isin(I))]
    # edges where source is internal and target is boundary
    e_i2 = edges[(edges['source'].isin(I)) & (edges['target'].isin(B))]
    
    e_i = pd.concat([e_i1, e_i2])
    e_i = e_i[e_i['gp_source'] == e_i['gp_target']]
    
    return e_i

def get_boundary_edges (edges, B):    
    '''
    Function to create set of edges linking nodes on boundaries within clusters
    
    Arguments:
        edges{DataFrame} -- dataset with the edges in the graph
        B{list} -- list with nodes on boundaries
    
    Returns:
        [DataFrame] -- edges of nodes on boundaries
    '''
    # edges where source and target are boundary from different clusters (modularity)
    e_b = edges[(edges['source'].isin(B)) & (edges['target'].isin(B))]
    e_b = e_b[e_b['gp_source'] != e_b['gp_target']]

    return e_b

def nodes_polarization (B, e_i, e_b):
    '''
    Function to calculate the polarization measure for each node on  boundary
    
    Arguments:
        B{list} -- list with nodes on boundaries
        e_i{DataFrame} -- dataset with internal edges
        e_b{DataFrame} -- dataset with boundary edges
    
    Returns:
        [dict] -- polarization of each node
    '''

    P = {x:None for x in B}
    
    for node in B:
        # number of links with internal nodes
        di = e_i[(e_i['source'] == node) | (e_i['target'] == node)].shape[0]
        # number of links with boundary nodes
        db = e_b[(e_b['source'] == node) | (e_b['target'] == node)].shape[0]
        
        p = di/(di+db)-.5
        P[node]   = p
        
    return P
        
def polarization (p_nodes):
    '''
    Function to calculate the polarization of the graph
    
    Arguments:
        p_nodes{dict} -- dictionary with nodes polarization
    
    Returns:
        [float] -- graph polarization measure
    '''    
    # we cannot say that there is a polarization measure when there is no boundary conections
    if len(p_nodes) == 0:
        p = np.nan
    else:
        p = np.mean([value for key, value in p_nodes.items()])
        
    return p