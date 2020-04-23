import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy
from scipy.sparse import csr_matrix
import scvelo as scv
#from bbknn import bbknn

sc.settings.verbosity = 3
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80)

class ClusteringWorkflow():

    def renameCR(
        self,
        mtx_folder
        ):

        """\
            Load and rename the cellranger output data

            Parameters
            ----------
            mtx_folder
                Required
                The folder containg the .mtx file to be read in,
            prepend
                Required
                String to prepend to cell barcodes
            -------
        """

        self.adataCR = sc.read_10x_mtx(
            path = mtx_folder,
            var_names = 'gene_symbols',
            cache = True
            )
        
        display(self.adataCR.obs)
        prepend = input('String to add to beginning of cell barcodes: ')
        self.adataCR.obs.index = prepend + self.adataCR.obs.index.str.replace('-1','')
    
    def renameV(
        self,
        loom_file
        ):

        """\
            Load and rename the velocyto ouput data

            Parameters
            ----------
            loom_file
                Required
                The file containg the .loom file to be read in,
            replace
                Required
                String to replace to 'possorted_genome_bam_GGCIQ:' with in cell barcodes
            -------
        """

        self.adataV = sc.read_loom(
            loom_file,
            var_names = 'Gene'
            )
        
        display(self.adataV.obs)
        remove = input('Phase to remove in cell barcodes: ')
        replace = input('Phrase to replace the removed phrase of cell barcodes: ')
        self.adataV.obs.index = self.adataV.obs.index.str.replace(remove,replace)
        self.adataV.obs.index = self.adataV.obs.index.str.replace('x','')

    def show_highest_expr_genes(
        self,
        dataset,
        save = False
        ):

        """\
            Visualize the highest expressed genes in data

            Parameters
            ----------
            dataset
                Required: {'cellranger','velocyto'}
                Specify which dataset to visualize the 20 highest expressed genes
            save
                Optional
                Whether to save the generated figure
            -------
        """

        if save == True:
            if dataset == 'cellranger':
                sc.pl.highest_expr_genes(self.adataCR, n_top=20, save = 'cell_ranger')
            elif dataset == 'velocyto':
                sc.pl.highest_expr_genes(self.adataV, n_top=20, save = 'velocyto')
        else:
            if dataset == 'cellranger':
                sc.pl.highest_expr_genes(self.adataCR, n_top=20)
            elif dataset == 'velocyto':
                sc.pl.highest_expr_genes(self.adataV, n_top=20)
    
    def filtering(
        self,
        dataset,
        minimum_genes = 700,
        minimum_cells = 8,
        ):

        """\
            Filter data based on user specification

            Parameters
            ----------
            dataset
                {'cellranger','velocyto'}
                Specify which dataset to perform calculations on
                
            minimum_genes
                Default = 700
                Minimum number of genes expressed required for a cell to pass filtering
            
            minimum_cells
                Default = 8
                Minimum number of cells a gene is expressed in to pass filtering
            -------
        """

        if dataset == 'cellranger':
            sc.pp.filter_cells(self.adataCR, min_genes = minimum_genes)
            sc.pp.filter_genes(self.adataCR, min_cells = minimum_cells)
            self.adataCR.obs['n_counts'] = self.adataCR.X.sum(axis=1).A1
            self.adataCR.obs['percent_mito'] = np.sum(self.adataCR[:, self.adataCR.var_names.str.startswith('mt-')].X, axis=1).A1 / np.sum(self.adataCR.X, axis=1).A1
            sc.pl.violin(self.adataCR, ['n_genes', 'n_counts', 'percent_mito'], jitter=0.4, multi_panel=True)
            sc.pl.scatter(self.adataCR, x='n_counts', y='percent_mito')
            sc.pl.scatter(self.adataCR, x='n_counts', y='n_genes')
            number_genes = float(input('Filter out cells with too many non-zero genes (> number_genes): '))
            number_counts = float(input('Filter out cells with greater than this many counts: '))
            mito_threshold = float(input('Remove cells with percentage mictochondrial genes greater than: '))
            self.adataCR = self.adataCR[self.adataCR.obs['n_genes'] < number_genes, :]
            self.adataCR = self.adataCR[self.adataCR.obs.percent_mito < mito_threshold, :]
            self.adataCR = self.adataCR[self.adataCR.obs['n_counts'] < number_counts, :]
        elif dataset == 'velocyto':
            sc.pp.filter_cells(self.adataV, min_genes = minimum_genes)
            sc.pp.filter_genes(self.adataV, min_cells = minimum_cells)
            self.adataV.obs['n_counts'] = self.adataV.X.sum(axis=1).A1
            self.adataV.obs['percent_mito'] = np.sum(self.adataV[:, self.adataV.var_names.str.startswith('mt-')].X, axis=1).A1 / np.sum(self.adataV.X, axis=1).A1
            sc.pl.violin(self.adataV, ['n_genes', 'n_counts', 'percent_mito'], jitter=0.4, multi_panel=True)
            sc.pl.scatter(self.adataV, x='n_counts', y='percent_mito')
            sc.pl.scatter(self.adataV, x='n_counts', y='n_genes')
            number_genes = int(input('Filter out cells with too many non-zero genes (> number_genes): '))
            number_counts = float(input('Filter out cells with greater than this many counts: '))
            mito_threshold = float(input('Remove cells with percentage mictochondrial genes greater than: '))
            self.adataV = self.adataV[self.adataV.obs['n_genes'] < number_genes, :]
            self.adataV = self.adataV[self.adataV.obs.percent_mito < mito_threshold, :]
            self.adataV = self.adataV[self.adataV.obs['n_counts'] < number_counts, :]
    
    def norm_log(
        self,
        dataset,
        counts_after
        ):

        """\
            Total-count normalize data and take natural log

            Parameters
            ----------
            dataset
                {'cellranger','velocyto'}
                Specify which dataset to perform calculations on
                
            counts_after
                Required
                Total count for each cell after normalization
            -------
        """

        if dataset == 'cellranger':
            sc.pp.normalize_per_cell(self.adataCR, counts_per_cell_after= counts_after)
            sc.pp.log1p(self.adataCR)
            self.adataCR.raw = self.adataCR
        elif dataset == 'velocyto':
            sc.pp.normalize_per_cell(self.adataV, counts_per_cell_after= counts_after)
            sc.pp.log1p(self.adataV)
            self.adataV.raw = self.adataV

    def highly_variable_genes(
        self,
        dataset,
        minimum_mean = 0.0125,
        maximum_mean = 3,
        minimum_disp = 0.5
        ):

        """\
            Identify Highly Vairbale Genes

            Parameters
            ----------
            dataset
                {'cellranger','velocyto'}
                Specify which dataset to perform calculations on
                
            minimum_mean
                Default = 0.0125

            maximum_mean
                Default = 3
            
            minimum_disp
                Default = 0.5
            -------
        """
        if dataset == 'cellranger':
            sc.pp.highly_variable_genes(self.adataCR, min_mean = minimum_mean, max_mean = maximum_mean, min_disp = minimum_disp)
            sc.pl.highly_variable_genes(self.adataCR)
        elif dataset == 'velocyto':
            sc.pp.highly_variable_genes(self.adataV, min_mean = minimum_mean, max_mean = maximum_mean, min_disp = minimum_disp)
            sc.pl.highly_variable_genes(self.adataV)
    
    def scale(
        self,
        dataset
        ):

        """\
            Scale data to unit variance

            Parameters
            ----------
            dataset
                {'cellranger','velocyto'}
                Specify which dataset to perform calculations on
            -------
        """

        if dataset == 'cellranger':
            sc.pp.scale(self.adataCR, max_value = 10)
        elif dataset == 'velocyto':
            sc.pp.scale(self.adataV, max_value = 10)
    
    def analysis(
        self,
        dataset,
        number_of_neighbors,
        show_pca_variance_plot = True,
        n_pcs = None
        ):

        """\
            Perform PCA, compute and embed the neighborhood graph and perform leid 

            Parameters
            ----------
            dataset
                {'cellranger','velocyto'}
                Specify which dataset to perform calculations on
            number_of_neighbors
                Required
                Number of neighbors to use for clustering
            show_pca_variance_plot
                Default = True, {True,False} 
                Whether to show the pca explained varaince ratio plot
            n_pcs
                Required if show_pca_variance_plot = False
                Number of PCs to use for clustering
            -------
        """

        if dataset == 'cellranger':
            sc.tl.pca(self.adataCR, svd_solver = 'arpack')
            if show_pca_variance_plot == False:
                n_pcsCR = n_pcs
            elif show_pca_variance_plot == True:
                sc.pl.pca_variance_ratio(self.adataCR, log = True)
                n_pcsCR = int(input('Use this number of PCs for clustering: '))
            sc.pp.neighbors(self.adataCR, n_neighbors = number_of_neighbors, n_pcs = n_pcsCR)
            sc.tl.umap(self.adataCR)
            sc.tl.leiden(self.adataCR)
            sc.pl.umap(self.adataCR, color = 'leiden', legend_loc = 'on data', title = '')
        elif dataset == 'velocyto':
            sc.tl.pca(self.adataV, svd_solver = 'arpack')
            if show_pca_variance_plot == False:
                n_pcsV = n_pcs
            elif show_pca_variance_plot == True:
                sc.pl.pca_variance_ratio(self.adataV, log = True)
                n_pcsV = int(input('Use this number of PCs for clustering: '))
            sc.pp.neighbors(self.adataV, n_neighbors = number_of_neighbors, n_pcs = n_pcsV)
            sc.tl.umap(self.adataV)
            sc.tl.leiden(self.adataV)
            sc.pl.umap(self.adataV, color = 'leiden', legend_loc = 'on data', title = '')