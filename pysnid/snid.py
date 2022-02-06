
""" Module to handle SNID fit. """

import os
import shutil
import numpy as np
import pandas
import warnings


def run_snid(filename, 
             phase=None, redshift=None, delta_phase=5, delta_redshift=None,
             lbda_range=[4000,8000], set_it=True,
             verbose=False, quiet=True, get_results=True, **kwargs):
    """ """
    snid_prop = dict(quiet=quiet, lbda_range=lbda_range, verbose=verbose)

    #
    # - Phase
    if phase is not None:
        snid_prop["phase_range"]=[phase-delta_phase, phase+delta_phase]
    #
    # - redshift            
    if redshift is not None:
        snid_prop["forcez"] = redshift
        if delta_redshift is not None:
            snid_prop["redshift_range"] = [redshift-delta_redshift, redshift+delta_redshift]


    # - Running SNID
    snidf = SNID()
    outfile = snidf.run(filename, **{**snid_prop,**kwargs})
    if outfile is None:
        warnings.warn("SNID fit failed. Nothing returned")
        return None

    if get_results:        
        snidres = SNIDReader.from_filename(outfile)
        return snidres
    
    return outfile


def bulk_run_snid(filenames, client=None, as_dask="delayed", map_kwargs={}, **kwargs):
    """ """
    import dask
    run_delayed = []
    for i,filename in enumerate(filenames):
        mkwargs = {k:v if not hasattr(v,"__iter__") else v[i] for k,v in kwargs.items()}
        print(mkwargs)
        run_delayed.append( dask.delayed(run_snid)(filename, **mkwargs))

    # ------------ #
    #   Dask Out   #
    # ------------ #
    if as_dask == "delayed":
        return run_delayed
    
    if as_dask == "compute":
        if client is None:
            return dask.delayed(list)(run_delayed).compute()
        return client.compute(run_delayed)
    
    if as_dask == "gather":
        if client is None:
            return dask.delayed(list)(run_delayed).compute()
        return client.gather( client.compute(run_delayed) )
    
    raise ValueError(f"as_dask can only delayed, compute and gather: {as_dask} given")
        
    


class SNIDReader( object ):

    def __init__(self, data=None, results=None, models=None):
        """ """
        if data is not None:
            self.set_data(date)
        if results is not None:
            self.set_results(results)
        if models is not None:
            self.set_models(models)

    @classmethod
    def from_filename(cls, filename):
        """ """
        this = cls()
        hdata = pandas.HDFStore(filename)
        filekeys = hdata.keys()
        if "/data" in filekeys:
            this.set_data( hdata.get("data") )
        else:
            warnings.warn(f"no 'data' stored in the input filename {filename}")
        if "/results" in filekeys:
            this.set_results( hdata.get("results") )
        else:
            warnings.warn(f"no 'results' stored in the input filename {filename}")
        
        if "/models" in filekeys:
            this.set_models( hdata.get( "models" ) )
        else:
            comps = [l for l in filekeys if "comp" in l]
            if len(comps)>0:
                warnings.warn("Important: Deprecation - the old '_snid.h5' format with individual 'comp file' stored will not be supported at the next upgrade. Rebuild your file.")
                this.set_models(pandas.concat({int(comp.split("comp")[-1]): hdata.get( comp ) for comp in comps}))
            else:
                warnings.warn(f"not a single 'comp' stored in the input filename {filename}")

        this._filename = filename
        return this


    @property
    def from_run(cls, filename, forcez=None, phase_range=[-20,50], redshift_range=[-0.01,0.4],
                     **kwargs):
        """ """
        raise NotImplementedError("To be implemented")
            
    # ============== #
    #  Method        #
    # ============== #
    def set_results(self, results):
        """ """
        self._results = results
        
    def set_data(self, data):
        """ """
        self._data = data
    
    def set_models(self, models):
        """ """
        self._models = models
    
    # --------- #
    #  GETTER   #
    # --------- #
    def get_model_label(self, index, incl_rlap=False):
        """ """
        mdata = self.results.loc[index]
        text = f"{mdata['type']} ({mdata['sn']}) @ z={mdata['z']:.3f} | phase={mdata['age']}"
        if incl_rlap:
            text+= f" | rlap={mdata['rlap']:.1f}"
        return  text
    
    def get_model_rlap(self, index):
        """ """
        return self.results.loc[index]["rlap"]

    def get_bestmatches(self, sortby="rlap"):
        """ """
        # The reset index is to have the no. columns in the returned
        # dataframe.
        results = self.results.sort_values(sortby, ascending=False).reset_index()
        bestmatches = results.groupby("type").first().sort_values("rlap", ascending=False)
        if "cutoff" in bestmatches.index:
            return bestmatches.drop("cutoff")
        return bestmatches

    def get_results(self, types="*", rlap_range=[5,None], 
                    lap_range=None, age_range=None, z_range=None):
        """ get a subset of the result dataframe """
        def _get_in_range_(res_, key, rmin=None, rmax=None):
            """ """
            if not rmin is None or not rmax is None:
                if rmax is None:
                    res_ = res_[res_[key]>=rmin]
                elif rmin is None:
                    res_ = res_[res_[key]<=rmax]
                else:
                    res_ = res_[res_[key].between(rmin, rmax)]
            return res_    


        res = self.results.copy()

        if not (types is None or types in ["*","all"]):
            if "*" in types:
                t_ = types.replace("*","")
                types = [t for t in res["type"].astype("str").unique() if t_ in t]
            else:
                types = np.atleast_1d(types)

            res = res[res["type"].isin(types)]

        if rlap_range is not None:
            res = _get_in_range_(res, "rlap", *rlap_range)
        if z_range is not None:
            res = _get_in_range_(res, "z", *z_range)
        if age_range is not None:
            res = _get_in_range_(res, "age", *age_range)

        if lap_range is not None:
            res = _get_in_range_(res, "lap", *lap_range)

        return res

    def get_inputdata(self, fluxcorr=True):
        """ For some reason, the 'data' spectra recorded by SNID (and
        insite self.data) corresponds to input_flux*input_lbda.
        fluxcorr enables to return the correct flux such that:
        
        input_flux = self.get_inputdata(fluxcorr=True)
        -> here input_flux is actually normalised by its mean.
        """
        data = self.data.copy()
        if not fluxcorr:
            return data

        flux = data["flux"]/data["wavelength"]
        flux/=flux.mean()
        data["flux"] = flux*1.05 # No idea why...
        return data

    def get_modeldata(self, model_, fluxcorr=True):
        """ """
        data = self.models.xs(model_).copy()
        if not fluxcorr:
            return data

        flux = data["flux"]/data["wavelength"]
        flux/=flux.mean()
        data["flux"] = flux*1.05 # No idea why...
        return data
        
    # --------- #
    #  GETTER   #
    # --------- #
    def show_bestmatches(self, nbest=None, ax=None, savefile=None, **kwargs):
        """ """
        best_matches = self.get_bestmatches()
        if nbest is not None:
            best_matches = best_matches.iloc[:nbest]
        # Limit to those with models.
        best_matches = best_matches[best_matches["no."].astype("int")<self.nmodels]
        models = np.asarray(best_matches["no."], dtype="int")

        return self.show(models=models, ax=ax, savefile=savefile, **kwargs)
    
        
    def show(self, models=[1], offset_coef=1, ax=None, savefile=None, fluxcorr=True,
                 lw_data=1.5, color_data="0.7", lw_model=1.5, modelprop={},
                 **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[7,4])
            ax = fig.add_axes([0.12,0.15,0.8,0.8])
        else:
            fig = ax.figure

        propmodel = {**dict(lw=lw_model),**modelprop}

        # - Data
        data_ = self.get_inputdata(fluxcorr=fluxcorr)
        
        for i, model_ in enumerate(np.atleast_1d(models)):
            datalabel = "snid-format data" if i==0 else "_no_legend_"
            offset = offset_coef*i
            ax.plot(data_["wavelength"], data_["flux"]-offset, 
                    label=datalabel, lw=lw_data, color=color_data, **kwargs)

            d = self.get_modeldata(model_, fluxcorr=fluxcorr)
            mlabel = self.get_model_label(str(model_))
            ax.plot(d["wavelength"], d["flux"]-offset, 
                    label=f"{model_}: {mlabel}", 
                    **propmodel)

            modeldata = self.results.loc[str(model_)]
            text = f"{modeldata['type']} \n z={modeldata['z']:0.3f} | {modeldata['age']:+0.1f}d \n  rlap: {modeldata['rlap']:0.1f} "
            ax.text(d["wavelength"][0]-50, d["flux"][0]-offset, text, 
                    va="center", ha="right", color=f"C{i}", 
                    fontsize="x-small", weight="bold")


        #ax.set_xlim(d["wavelength"][0]*0.92)
        #ax.legend(frameon=False, fontsize='x-small')
        ax.set_yticks([])

        clearwhich = ["left","right","top"] # "bottom"
        [ax.spines[which].set_visible(False) for which in clearwhich]

        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize="large")
        if savefile is not None:
            fig.savefig(savefile)
            
        return fig
    # ============== #
    #  Internal      #
    # ============== #    
    @staticmethod
    def _read_snidflux_(filename_):
        """ """
        data = [l.split() for l in open(filename_).read().splitlines() if not l.strip().startswith("#")]
        columns = ["wavelength", "flux"]
        return pandas.DataFrame(np.asarray(data, dtype="float"), columns=columns)
        
    @staticmethod
    def _read_snidoutput_(filename_, nfirst=None):
        """ """
        f = open(filename_).read().split("### rlap-ordered template listings ###")[-1].splitlines()
        dd = pandas.DataFrame([l.split() for l in f[2:]], columns=f[1][1:].split()).set_index("no.")
        dd = dd[~dd["age_flag"].isin(["cut"])] # safeout
        if nfirst is not None:
            dd = dd.iloc[:nfirst]
            
        return dd.astype({**{k:"str" for k in ["sn","type","grade"]},
                                   **{k:"float" for k in ["lap","rlap","z","zerr","age"]},
                                     **{k:"bool" for k in ["age_flag"]}}
                                   )

    
    # ============== #
    #  Properties    #
    # ============== #    
    @property
    def data(self):
        """ """
        if not hasattr(self,"_data"):
            return None
        return self._data

    @property
    def models(self):
        """ """
        if not hasattr(self,"_models"):
            return None
        return self._models
    
    @property
    def results(self):
        """ """
        if not hasattr(self,"_results"):
            return None
        return self._results

    @property
    def filename(self):
        """ """
        if not hasattr(self,"_filename"):
            return None
        return self._filename

    @property
    def nmodels(self):
        """ number of model stored inside self.models """
        return len(self.models.index.levels[0])


class SNID( object ):
    """ """
    def __init__(self, id_=None):
        """ """
        if id_ is None:
            self._snidid = f"{np.random.randint(1000000):08d}"
        else:
            self._snidid = f"{id_}"
        
    @staticmethod
    def build_snid_command(filename, 
                            forcez=None,
                            lbda_range=[4000,8000], 
                            phase_range=[-20,50],
                            redshift_range=[-0.01,0.4],
                            medlen=20, fwmed=None,
                            rlapmin=2, 
                            fluxout=30,
                            skyclip=False, aband=False, inter=False, plot=False,
                            param=None):
        """ """
        lbdamin, lbdamax = lbda_range
        agemin, agemax = phase_range
        zmin, zmax = redshift_range
        
        cmd_snid  = f"snid "
        if param is not None:
            cmd_snid += f"param={param} "
            
        cmd_snid += f"wmin={int(lbdamin)} wmax={int(lbdamax)} "
        # Redshift
        if forcez is not None:
            cmd_snid += f"forcez={forcez} "
        cmd_snid += f"zmin={zmin} zmax={zmax} "
        # Phase
        cmd_snid += f"agemin={agemin} agemax={agemax} "
        # Input Spectral Structure
        cmd_snid += f"skyclip={int(skyclip)} " 
        if medlen is not None:
            cmd_snid += f"medlen={int(medlen)} " 
        if fwmed is not None:
            cmd_snid += f"fwmed={int(fwmed)} " 
            
        cmd_snid += f"fluxout={int(fluxout)} aband={int(aband)} rlapmin={int(rlapmin)} inter={int(inter)} plot={int(plot)} "
        cmd_snid += f"{filename}"
        print(cmd_snid)
        return cmd_snid
    
    def run(self, filename, fileout=None,
                dirout=None, tmpdir=None,
                cleanout=True, verbose=False,
                quiet=False, paramfile=None, in_tmpdir=True,
                **kwargs):
        """ run SNID and store the result as a hdf5 file. 
        
        **kwargs goes to build_snid_command
        forcez=None,
        lbda_range=[4000,8000], 
        phase_range=[-20,30],
        redshift_range=[0,0.2],
        medlen=20, rlapmin=4, 
        fluxout=30,
        skyclip=False, aband=False, inter=False, plot=False
        
        """
        from subprocess import PIPE, run
        from glob import glob
        #
        basename = os.path.basename(filename)
        dirname  = os.path.dirname(filename)        
        #
        # Create a symlink to bypass the SNID filepath limitation
        
        if tmpdir is None:
            tmpdir = f"tmpsnid_{self._snidid}"
        if not os.path.isdir(tmpdir):
            os.makedirs(tmpdir, exist_ok=True)
        if in_tmpdir:
            old_pwd=os.getcwd()
            os.chdir(tmpdir)
            self._tmpfile = f"snid_{self._snidid}_spectofit.ascii"
        else:
            old_pwd = None
            self._tmpfile = os.path.join(tmpdir, f"snid_{self._snidid}_spectofit.ascii")
            
        os.symlink(filename, self._tmpfile)

        tmpbase = os.path.basename(self._tmpfile).split(".")[0]
        
        snid_cmd = self.build_snid_command(self._tmpfile, param=paramfile, **kwargs)
        
        self._result = run(snid_cmd.split(), stdout=PIPE, stderr=PIPE, universal_newlines=True)
        if verbose:
            print(f" running: {snid_cmd}")
            print(self._result.stdout.split("\n"))
        
        if self._result.returncode != 0:
            warnings.warn("SNID returncode is not 0, suggesting an error")
        elif "orrelation function is all zero!" in self._result.stdout:
            warnings.warn("SNID failed:  Searching all correlation peaks... PEAKFIT: Correlation function is all zero!")
        elif "PEAKFIT: fit quits before half peak points!" in self._result.stdout:
            warnings.warn("SNID failed:  Searching all correlation peaks... PEAKFIT: fit quits before half peak points!")
        else:
            datafile = f"{tmpbase}_snidflux.dat"
            modelfiles = glob(f"{tmpbase}_comp*_snidflux.dat")
            snidout = f"{tmpbase}_snid.output"
            try:
                result = SNIDReader._read_snidoutput_(snidout)
            except FileNotFoundError:
                print(" SNID RETURN CODE ".center(40,"-"))
                print(self._result.stdout)
                print("".center(40,"-"))
                if cleanout: self._cleanup_run_(tmpdir, old_pwd=old_pwd)
                raise FileNotFoundError("cannot find the SNID output.")

            data = SNIDReader._read_snidflux_(datafile)
            models = pandas.concat({int(f_.split("comp")[-1].split("_")[0]):SNIDReader._read_snidflux_(f_) 
                                        for i,f_ in enumerate(modelfiles)})
            
            if fileout is None:
                if dirout is None:
                    dirout = dirname
                fileout = os.path.join(dirout,basename.split(".")[0]+"_snid.h5")
                
            elif not fileout.endswith("h5"):
                fileout+=".h5"
                
            result.to_hdf(fileout, key="results")
            data.to_hdf(fileout, key="data")
            models.to_hdf(fileout, key="models")
                
            if not quiet:
                print(f"snid run was successfull: data stored at {fileout}")

            if cleanout:
                _ = os.remove(snidout)
                _ = os.remove(datafile)
                _ = [os.remove(f_) for f_ in modelfiles]
                
        # - cleanup
        if cleanout:
            self._cleanup_run_(tmpdir,
                                   old_pwd=old_pwd)

        return fileout

    def _cleanup_run_(self, tmpdir, old_pwd=None):
        """ """
        os.remove("snid.param")
        os.remove(self._tmpfile)
        if old_pwd is not None:
            os.chdir(old_pwd)
            
        shutil.rmtree(tmpdir)
    # ============== #
    #  Internal      #
    # ============== #    
    def _build_tmpfile_(self, tmpdir="tmp", tmpstruct="_default_"):
        """ """
        if not os.path.isdir(tmpdir):
            os.makedirs(tmpdir, exist_ok=True)
            
        if tmpstruct == "_default_":
            tmpstruct = f"snid_{self._snidid}_spectofit"
            
        tmp_file = os.path.join(tmpdir, tmpstruct+".ascii")
        i=1
        while os.path.isfile(tmp_file):
            tmp_file = os.path.join(tmpdir, tmpstruct+f"_{i}"+".ascii")
            i+=1
            
        return tmp_file
    
