"""Configured HPO entry point; no computation occurs until main() is invoked.

Example: python examples/compound_hyperopt.py --kernel K.npy --traits Y.csv
         --checkpoint study_dir --trials 30
Rows/columns of K must match the labeled phenotype rows. CSV first column is
the row identifier; optional trait-project CSV has trait,project columns.
"""
from pathlib import Path
import argparse


def main(argv=None):
    """Fit the selected full-training covariance with supervised HPO.

    argv is an optional argument list, default process arguments. Returns the
    labeled fit. Cost is the configured fold/trial workload and final fit; this
    example does not execute tests or claim optimal resource defaults.
    """
    import numpy as np
    import pandas as pd
    import vigor
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',required=True)
    parser.add_argument('--traits',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--trait-projects')
    parser.add_argument('--sample-metadata')
    parser.add_argument('--trials',type=int,default=30)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--h2-prior',choices=['off','search'],default='search')
    args=parser.parse_args(argv)
    Y=pd.read_csv(args.traits,index_col=0);Y.index=Y.index.astype(str)
    K=np.load(args.kernel,mmap_mode='r',allow_pickle=False)
    cv={'designs':['unseen_animal','partial_cell'],'inner_folds':2}
    if args.trait_projects:
        project=pd.read_csv(args.trait_projects,dtype=str).set_index('trait')['project']
        cv.update(trait_projects=project,designs=['unseen_animal','partial_cell','few_shot_project','donor_ablation'])
    if args.sample_metadata:
        metadata=pd.read_csv(args.sample_metadata,index_col=0);metadata.index=metadata.index.astype(str)
        cv['sample_metadata']=metadata
    space={} if args.h2_prior=='search' else {'h2_prior_strength':[0.]}
    fit=vigor.genetic_varcov_PSD(K,Y,hyper_opt=True,analytic_se=False,n_boot=0,
        blas_threads=args.threads,hyper_opt_kwargs={'cv':cv,'n_trials':args.trials,
            'seed':args.seed,'checkpoint_dir':str(Path(args.checkpoint).resolve()),'resume':args.resume,
            'search_space':space})
    print({'status':fit.attrs['hpo_status'],'final_fit':fit.attrs['hpo_final_refit_complete'],
           'selected':fit.attrs['hpo_selected_params'],'model':fit.attrs.get('hpo_model_path'),
           'checkpoint':fit.attrs['hpo_checkpoint']})
    return fit


if __name__=='__main__':
    main()
