import pandas as pd
import numpy as np
import itertools
import logging
from helpers import parser
import helpers as fc


if __name__ == '__main__':

    args = parser.parse_args()
    
    path = str(args.path)
    log_path = str(args.log_file)
    cluster_ids = args.clusters

    logging.basicConfig(filename = log_path+'/polarization.log',level=logging.INFO)
    logger = logging.getLogger()

    groups = pd.read_csv(path+'nodes.csv', names = ['name','group'], header = 0, sep=',')
    edges = pd.read_csv(path+'edges.csv', names = ['source','target'], header = 0, sep = ',')
    edges = edges.drop_duplicates()


    logger.info('Filter of relevant clusters')
    edges, groups = fc.filter_graph(edges, groups, cluster_ids)

    pairs = itertools.combinations(np.unique(groups['group']),2)
    
    result = []
    for _id, (a, b) in enumerate(pairs):
        p_edges = edges[(edges['gp_source'].isin([a,b])) & (edges['gp_target'].isin([a,b]))]
        
        logger.info('Finding internal nodes')
        internal_nodes = fc.get_internal(p_edges)
        
        logger.info('Finding nodes on boundary')
        boundary_nodes = fc.get_boundaries(p_edges, groups, internal_nodes)
    
        logger.info('Creating set of edges with internal nodes')
        internal_edges = fc.get_internal_edges(p_edges, internal_nodes, boundary_nodes)
        
        logger.info('Creating set of edges with nodes on boundary')
        boundary_edges = fc.get_boundary_edges(p_edges, boundary_nodes)
        
        logger.info('Calculation of nodes polarization')
        nodes_p = fc.nodes_polarization(boundary_nodes, internal_edges, boundary_edges)
    
        logger.info('Calculation of polarization for clusters ' + str(a) + ' e '+ str(b))
        p = fc.polarization(nodes_p)
        
        result.append([a,b,p])

    data = pd.DataFrame(result)
    data.to_csv(f'path/clusters_polarization.csv')