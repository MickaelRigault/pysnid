
""" Module to handle SNID fit. """

import os
import shutil
import numpy as np
import pandas
import warnings


def run_snid(filename,
             phase=None, redshift=None, delta_phase=5, delta_redshift=None,
             redshift_bounds=[0, None],
             lbda_range=[4000,8000], set_it=True,
             verbose=False, quiet=True, get_results=True,
             rm_zeros=True, **kwargs):
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
            if redshift_bounds[0] is not None:
                min_redshift = np.max([redshift-delta_redshift, redshift_bounds[0]])
            else:
                min_redshift = redshift-delta_redshift

            if redshift_bounds[1] is not None:
                max_redshift = np.min([redshift+delta_redshift, redshift_bounds[1]])
            else:
                max_redshift = redshift+delta_redshift

            snid_prop["redshift_range"] = [min_redshift, max_redshift]
    else:

        snid_prop["redshift_range"] = redshift_bounds

    print(snid_prop)
    # - Running SNID
    snidf = SNID()
    options = {**snid_prop,**kwargs}
    if verbose:
        print("options used:", options)

    outfile = snidf.run(filename, **options)
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
    def from_filename(cls, filename, load_data=True, load_models=True, load_results=True):
        """ """
        this = cls()
        hdata = pandas.HDFStore(filename)
        filekeys = hdata.keys()
        if load_data:
            if "/data" in filekeys:
                this.set_data( hdata.get("data") )
            else:
                warnings.warn(f"no 'data' stored in the input filename {filename}")
        if load_results:
            if "/results" in filekeys:
                this.set_results( hdata.get("results") )
            else:
                warnings.warn(f"no 'results' stored in the input filename {filename}")
        if load_models:
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
        hdata.close()
        return this

    @property
    def from_run(cls, filename,
                     **kwargs):
        """ """
        raise NotImplementedError("To be implemented")

    # ============== #
    #  Method        #
    # ============== #
    def set_results(self, results):
        """ """
        results = results.copy()
        types = results["type"]
        results[["typing", "subtyping"]] = results["type"].str.split("-", n=1, expand=True)
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

    def get_bestmatches(self, sortby="rlap", grade="good", rlap_range=[4,None], **kwargs):
        """ grade="good" and **kwargs goes to get_result() """
        # The reset index is to have the no. columns in the returned
        # dataframe.
        results = self.get_results(grade=grade,rlap_range=rlap_range, **kwargs).sort_values(sortby, ascending=False).reset_index()
        bestmatches = results.groupby("type").first()
        size_ = results.groupby("type").size()
        size_.name="nentries"
        bestmatches = pandas.merge(bestmatches, size_, left_index=True, right_index=True
                                       ).sort_values("rlap", ascending=False)

        if "cutoff" in bestmatches.index:
            return bestmatches.drop("cutoff")
        return bestmatches

    def get_results(self, types="*", typing=None,
                        rlap_range=None,
                        lap_range=None, age_range=None, z_range=None,
                        grade="good", nfirst=30):

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

        # Go
        if grade is None:
            res = self.results
        else:
            res = self.results[self.results["grade"] == grade]

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

        #
        # = Typing
        #
        if typing in ["*","all", "any"]:
            typing = None

        if typing in ["snia", "sn ia", "Ia"]:
            typing = res[res["typing"] == "Ia"]["type"].unique()

        elif typing in ["Ia!norm"]:
            typing = res[(res["typing"] == "Ia") & (res["type"] != "Ia-norm")]["type"].unique()

        elif typing is not None:
            if typing == "auto":
                auto_type = self.get_type()
                typing_, subtype_ = np.transpose(auto_type)[0]

                if typing_ == "unclear":
                    typing = None
                elif subtype_ == "unclear":
                    typing = bestres[bestres["typing"] == "Ia"]["type"].unique()
                else:
                    typing = f"{typing_}-{subtype_}"

        # else typing is None
        if typing is not None:
            res = res[res["type"].isin(np.atleast_1d(typing))]
        # ============ #

        if nfirst is not None:
            res = res.iloc[:nfirst]

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

    def get_type(self, min_rlap=5, nfirst=10, grade='good', min_prob=0.5,
                 full_output=False, fallback='unclear', min_probsub=None,
                 incl_subtype=True,
                   **kwargs):
        """
        min_probsub -> min_prob is None

        If incl_subtype, the subtyping is the subtyping given the best_typing.
        So it reads as: p(subtype | besttype)

        Returns
        -------
        (string, float), [(string, float) if uncl_subtype]

        in details: (typename, p(type)), (subtype, p(subtype | type))

        """
        results = self.get_results(grade=grade,nfirst=nfirst, **{**dict(rlap_range=[min_rlap,None]), **kwargs})

        rlap_sums = results.groupby(["typing","subtyping"])["rlap"].sum()
        rlap_sums /= rlap_sums.sum()
        rlap_sums = rlap_sums.sort_values(ascending=False)

        if full_output:
            return rlap_sums

        if len(rlap_sums)==0:
            warnings.warn(f"No 'rlap' greater than {min_rlap} ; '{fallback}'  returned")
            best_type, best_typefrac = (fallback, np.nan)
        else:
            typing_frac = rlap_sums.groupby(level=0).sum().sort_values(ascending=False) # already a frac
            if typing_frac.iloc[0] < min_prob: # because stored by ascending=False
                warnings.warn(f"No 'probabilities' above the {min_prob:.0%} ; '{fallback}' typing returned")
                best_type, best_typefrac = (fallback, np.nan)
            else:
                best_typing = typing_frac.iloc[0] # because stored by ascending=False
                best_type, best_typefrac = typing_frac.index[0],typing_frac.iloc[0]

        if not incl_subtype:
            return (best_type, best_typefrac)

        if best_type == fallback:
            return (best_type, best_typefrac), (fallback, np.nan)

        # Subtyping:
        if min_probsub is None:
            min_probsub = min_prob

    #    return (best_type, best_typefrac), rlap_sums
        subtypes = rlap_sums.xs(best_type)
        subtype_frac = (subtypes/subtypes.sum()).sort_values(ascending=False)

        if subtype_frac.iloc[0]<min_prob:
            return (best_type, best_typefrac), (fallback, np.nan)
        return (best_type, best_typefrac), (subtype_frac.index[0],subtype_frac.iloc[0])

    def get_typing_result(self, typing="auto",
                         rlap_range=[5,None], nfirst=30):
        """ """
        bestres = self.get_results(rlap_range=rlap_range,nfirst=nfirst)

        return bestres

    def get_redshift(self, typing="auto", weight_by="rlap",
                        rlap_range=[5, None], nfirst=30,
                        dredshift="nmad",
                        default_zerr=np.nan, **kwargs):
        """

        Parameters
        ----------
        typing: [None, string or list of] -optional-
            if None or "*" all types will be used.
            if auto, typing comes from auto-typing (self.get_type())
            if given (could be a list), only given type is used.
            if "Ia" all subtypes

        """
        DEFAULT = [np.nan, default_zerr]
        bestres = self.get_results(typing=typing, nfirst=nfirst,
                                    rlap_range=rlap_range, **kwargs)

        # nothing so back to nan
        if len(bestres) == 0:
            return DEFAULT


        if weight_by is not None:
            weights = bestres[weight_by]

        # Redshift
        redshift = np.average(bestres["z"], weights=weights)

        # Error on
        if dredshift == "nmad":
            if len(bestres) < 3:
                dredshift = default_zerr
            else:
                from scipy.stats import median_abs_deviation
                dredshift = median_abs_deviation(bestres["z"])
        else:
            dredshift = getattr(np,"dredshift")(bestres["z"])

        return redshift, dredshift


    # --------- #
    #  GETTER   #
    # --------- #
    def show(self, axes=None, fig=None,
                 nbest=4,min_rlap=5,nfirst_resummary=30,
                 show_typing=True, label=None,
                 sumprop={},
                 redshift=None, zlabel=None, phase=None, dphase=None,
                 savefile=None,
                 show_telluric=True, show_ha=True, telluric_color="0.6",
                 **kwargs):
        """ """

        if axes is None:
            if fig is None:
                import matplotlib.pyplot as mpl
                # fig = mpl.figure(figsize=[9,3])
                fig = mpl.figure(figsize=[9,5])

            axs = fig.add_axes([0.1,0.18,0.55,0.7])
            #axt = fig.add_axes([0.75,0.1,0.2,0.75], polar=True)

            axsum= [fig.add_axes([0.75,0.82, 0.2,0.03]),
                    fig.add_axes([0.75,0.18, 0.2,0.55])]

        else:
            axs, axsum = axes
            fig = axs.figure

        _ = self.show_bestmatches(ax=axs, nbest=nbest, min_rlap=min_rlap, **kwargs)
        _ = self.show_ressummary(axes=axsum, nfirst=nfirst_resummary, min_rlap=min_rlap,
                                 redshift=redshift, zlabel=zlabel, phase=phase, dphase=dphase,
                                 **sumprop)

        # - Show typing
        if show_typing:
            typing, subtyping = self.get_type(incl_subtype=True)
            if label is not None:
                axs.text(-0.05, 1.1, f"{label}",
                         va="bottom", ha="left", fontsize="small", weight="normal",
                         color="0.6", transform=axs.transAxes)

            fig.text(-0.05, 1.01, f"auto typing: p({typing[0]})={typing[1]:.0%} | p({subtyping[0]}|{typing[0]})={subtyping[1]:.0%}",
                     va="bottom", ha="left", fontsize="small", weight="normal",
                     color="k", transform=axs.transAxes)

        if show_ha and redshift is not None:
            axs.axvline(6563*(1+redshift),  ls="--",  color=telluric_color, zorder=1, lw=0.5, alpha=0.8)

        if show_telluric:
            main_telluric = [7450,7750]
            small_telluric = [6850,7050]
            axs.axvspan( *main_telluric, color=telluric_color, alpha=0.05)
            axs.axvspan( *small_telluric, color=telluric_color, alpha=0.05)

        if savefile is not None:
            fig.savefig(savefile)

        return fig

    def show_spider(self, ax=None, main="rlap", second="nentries",
                        logscale="second", nfirst=None,
                        min_rlap=5, matchprop={},
                        color_main="C0", falpha_main=0.05, lw_main=1.5,
                        color_second="C1",falpha_second=0.2, lw_second=1,
                        color_grid=None, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        from matplotlib.colors import to_rgba
        from pysnid.tools import make_spiderplot, get_polartwin

        logscale = np.atleast_1d(logscale) if logscale is not None else []
        best_matches = self.get_bestmatches(**{**dict(rlap_range=[min_rlap,None]), **matchprop},nfirst=nfirst)
        # if nfirst is not None:
        #     best_matches = best_matches.iloc[:nfirst]

        nbest_matchs = len(best_matches)
        if nbest_matchs == 0:
            warnings.warn("No bestmatch")
            return None

        # - Labels
        labels = np.asarray(best_matches.index, dtype=str)
        # - Main axis
        valuemain = np.asarray(best_matches[main], dtype="float")
        if "main" in logscale:
            valuemain = np.log10(valuemain)
        # - Secondx
        if second is not None:
            valuesecond = np.asarray(best_matches[second], dtype="float")
            if "second" in logscale:
                valuesecond = np.log10(valuesecond)
        else:
            valuesecond = None

        fig = make_spiderplot(valuemain, labels=labels, ax=ax,
                              gcolor=color_grid,
                              rlabel_angle= (360/nbest_matchs)/2,
                              facecolor=to_rgba(color_main,falpha_main),
                              edgecolor=color_main, lw=lw_main,
                              rlabel=main, zorder=8,
                              highlight=min_rlap, highlight_color=color_grid)
        if ax is None:
            ax = fig.axes[0]

        if valuesecond is not None:
            fig = make_spiderplot(valuesecond, rtwin_from=ax,
                                  labels=labels, gcolor="None",
                                  facecolor=to_rgba(color_second, falpha_second),
                                  edgecolor=color_second,
                                  rlabel_angle=360/nbest_matchs * (0.5+int(nbest_matchs/2)),
                                  lw=lw_second, #alpha=1,
                                  rlabel=f"log(n>{min_rlap})" if second=="nentries" else second,
                                  rlabel_rotation=0, rlabel_ha="center", zorder=9)

        return fig


    def show_ressummary(self, axes=None, nfirst=30,
                        phase=None, dphase=None,
                        redshift=None, zlabel=None,
                        min_rlap=5, phase_prior=5,
                        line_color="0.6", resprop={}):
        """ """
        from matplotlib.colors import to_rgba
        if axes is None:
            import matplotlib.pyplot as mpl
            fig = mpl.figure(figsize=[4,4])
            ax  = fig.add_axes([0.175,0.85, 0.725,0.03])
            axr = fig.add_axes([0.175,0.20, 0.725,0.55])
        else:
            ax, axr = axes
            fig = ax.figure

        # res = self.get_results(rlap_range=[min_rlap,None], **resprop).iloc[:nfirst]
        res = self.get_results(rlap_range=[min_rlap,None],nfirst=nfirst, **resprop)
        #
        # - Rankind
        #
        types_values = res.reset_index().groupby("type")["no."].apply(list).sort_values()
        # Loop over groups
        for i_, (name, nos) in enumerate(types_values.items()):
            color = f"C{i_}"
            for j_, v_ in enumerate(np.asarray( list(nos), dtype="int")):
                if j_==0:
                    label= f"{name}: #{v_}"
                else:
                    label= "_no_legend_"

                ax.axvspan(v_-1,v_, facecolor=color, edgecolor="0.7", lw=0.5, label=label)
                if j_==0:
                    ax.text(v_-0.5, -0.8, f"{res.loc[str(v_)]['rlap']:.1f}",
                           color=color, va="top", ha="center", fontsize="x-small")
        # - ranking legend
        try:
            ax.legend(loc=[0,1.2],
                  ncol=np.min([2,len(types_values)]), fontsize="x-small", frameon=False,
                 columnspacing=2, handlelength=1)
        except:
            warnings.warn("legend failed")

        _ = ax.set_xlim(0,nfirst)#len(res))
        _ = ax.set_yticks([])
        _ = ax.set_xticks([])
        # - Ranking texts
        ax.text(-0.08,   0.3, "rank", va="center", ha="right",
                    fontsize="x-small", color="0.6",
                    transform=ax.transAxes)
        ax.text(-0.08, -1.3,  "rlap", va="center", ha="right",
                    fontsize="x-small", color="0.6",
                    transform=ax.transAxes)

        if len(res)<nfirst-1:
            ax.text(len(res), 0.3, f"#{len(res)}",
                    va="center", ha="left", fontsize="xx-small", color="0.3")
        ax.text(1.01, 0.3, f"#{nfirst}", va="center", ha="left", fontsize="x-small", color="0.3",
               transform=ax.transAxes)

        #
        # - Scatter
        #

        for i_,(name, nos) in enumerate(types_values.items()):
            typeres = res.loc[nos]
            axr.scatter(typeres["age"], typeres["z"],
                            facecolors=to_rgba(f"C{i_}", 0.9),
                            edgecolors="0.7", lw=0.5, zorder=3)

        if phase is not None:
            if dphase is not None:
                axr.axvspan(phase-2*dphase, phase+2*dphase, color=to_rgba(line_color,0.1),
                                zorder=2)
                axr.axvspan(phase-3*dphase, phase+3*dphase, color=to_rgba(line_color,0.1),
                                zorder=2)
            else:
                axr.axvline(phase, color=line_color, lw=1)

            axr.axvspan(phase-phase_prior,phase+phase_prior, zorder=1,
                            color=to_rgba("C0",0.02))


        if redshift is not None:
            fig.canvas.draw()
            minx_axr = axr.transData.inverted().transform(axr.transAxes.transform([0,-1]))[0]

            propz = dict(color=line_color, lw=1, ls="-")
            axr.axhline(redshift, **propz)
            if zlabel is not None:
                propzsource = dict(va="bottom", ha="left", color="0.6", fontsize="x-small")
                axr.text(minx_axr, redshift, zlabel, **propzsource)

        # axr.set_xlim(phase-7*dphase,phase+7*dphase)
        # axr.set_xlim(np.nanmin(typeres["age"])-3,np.nanmax(typeres["age"])+3)

        axr.set_ylabel("Redshift", fontsize="small")
        axr.set_xlabel("Phase", fontsize="small")
        axr.tick_params(labelsize="small")
        return fig

    def show_bestmatches(self, nbest=None, ax=None, savefile=None, min_rlap=5, matchprop={}, **kwargs):
        """ """
        best_matches = self.get_bestmatches(**{**dict(rlap_range=[min_rlap,None]), **matchprop},nfirst=nbest)
        # if nbest is not None:
        #     best_matches = best_matches.iloc[:nbest]
        # Limit to those with models.
        best_matches = best_matches[best_matches["no."].astype("int")<self.nmodels]
        models = np.asarray(best_matches["no."], dtype="int")

        return self.show_models(models=models, ax=ax, savefile=savefile, **kwargs)

    def show_models(self, models=[1], offset_coef=None, ax=None, savefile=None, fluxcorr=True,
                 lw_data=1.5, color_data="0.7", lw_model=1.5, modelprop={},
                 **kwargs):
        """
        offset_coef: [float, 'None' or None]
            offset applied between spectra.
            - float: value used
            - None : go back to default values (90% of data)
            -'None': No offset ; same as offset_coef=0


        """
        if ax is None:
            from matplotlib.figure import Figure
            fig = Figure(figsize=[7,4])
            ax = fig.add_axes([0.12,0.15,0.8,0.8])
        else:
            fig = ax.figure

        propmodel = {**dict(lw=lw_model),**modelprop}

        # - Data
        data_ = self.get_inputdata(fluxcorr=fluxcorr)
        if offset_coef is None:
            offset_coef = np.percentile(data_["flux"], 90)*0.8
        elif offset_coef == "None":
            offset_coef = 0

        models = np.atleast_1d(models)

        if len(models)==0:
            ax.plot(data_["wavelength"], data_["flux"],
                    lw=lw_data, color=color_data, **kwargs)

        for i, model_ in enumerate(models):
            datalabel = "snid-format data" if i==0 else "_no_legend_"
            offset = offset_coef*i
            ax.plot(data_["wavelength"], data_["flux"]-offset,
                    label=datalabel, lw=lw_data, color=color_data, **kwargs)

            d = self.get_modeldata(model_, fluxcorr=fluxcorr)
            mlabel = self.get_model_label(str(model_))
            modeldata = self.results.loc[str(model_)]
            if modeldata['grade'] != "good":
                propmodel["ls"] = ":"

            ax.plot(d["wavelength"], d["flux"]-offset,
                    label=f"{model_}: {mlabel}",
                    **propmodel)


            text = f"{modeldata['sn']}: {modeldata['type']} \n z={modeldata['z']:0.3f} | {modeldata['age']:+0.1f}d \n  rlap: {modeldata['rlap']:0.1f}"
            if modeldata['grade'] != "good":
                text += f" ({modeldata['grade']})"
            ax.text(d["wavelength"][0]-50, d["flux"][0]-offset, text,
                    va="center", ha="right", color=f"C{i}",
                    fontsize="x-small", weight="bold")


        #ax.set_xlim(d["wavelength"][0]*0.92)
        #ax.legend(frameon=False, fontsize='x-small')
        ax.set_yticks([])

        clearwhich = ["left","right","top"] # "bottom"
        [ax.spines[which].set_visible(False) for which in clearwhich]

        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize="medium")
        ax.tick_params(labelsize="small")
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
        with open(filename_) as file:
            lines = file.read().split(
                "### rlap-ordered template listings ###"
            )[-1].splitlines()

        columns = lines[1].lstrip("#").split()

        rows=[]
        for line in lines[2:]:
            if line.lstrip().startswith("#"):
                break
            if line.strip():
                rows.append(line.split())

        dd = pandas.DataFrame(rows, columns=columns).set_index("no.")

        # Remove matches below the SNID rlap cutoff.
        # dd = dd[dd["grade"] != "cut"]

        if nfirst is not None:
            dd = dd.iloc[:nfirst]

        return dd.astype({
            "sn": "str",
            "type": "str",
            "grade": "str",
            "lap": "float",
            "rlap": "float",
            "z": "float",
            "zerr": "float",
            "age": "float",
            "age_flag": "int",
        })

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
                            lbda_range=[4000,9000],
                            phase_range=[-90,1000],
                            redshift_range=[-0.05,0.4],
                            use_sn_template=None,use_sn_type=None,
                            fwmed=None,
                            medlen=None,
                            rlapmin=2,
                            fluxout=30,
                            emclip=None,emwid=40,
                            skyclip=False, aband=False, inter=False, plot=False,
                            param=None, verbose=True):
        """ """

        print("*** build_snid_command ***")

        cmd_snid  = f"snid "
        if param is not None:
            cmd_snid += f"param={param} "

        if lbda_range is not None:
            lbdamin, lbdamax = lbda_range
            cmd_snid += f"wmin={int(lbdamin)} wmax={int(lbdamax)} "

        # Redshift
        if forcez is not None:
            cmd_snid += f"forcez={forcez} "

        if redshift_range is not None:
            zmin, zmax = redshift_range
            cmd_snid += f"zmin={zmin} zmax={zmax} "

        # Phase
        if phase_range is not None:
            agemin, agemax = phase_range
            cmd_snid += f"agemin={agemin:.0f} agemax={agemax:.0f} "

        # use of specific templates or types/subtypes
        if use_sn_template is not None:
            cmd_snid += f"use={use_sn_template} "
        if use_sn_type is not None:
            cmd_snid += f"usetype={use_sn_type} "

        #emclip/emwid
        if emclip is not None:
            cmd_snid += f"emclip={emclip} "
        if emwid is not None:
            cmd_snid += f"emwid={emwid} "

        # Input Spectral Structure
        cmd_snid += f"skyclip={int(skyclip)} "
        if medlen is not None:
            cmd_snid += f"medlen={int(medlen)} "
        if fwmed is not None:
            cmd_snid += f"fwmed={int(fwmed)} "

        cmd_snid += f"fluxout={int(fluxout)} aband={int(aband)} rlapmin={int(rlapmin)} inter={int(inter)} plot={int(plot)} "
        cmd_snid += f"{filename}"
        if verbose:
            print(cmd_snid)

        return cmd_snid

    def run(self, filename, fileout=None,
        dirout=None, tmpdir=None,
        cleanout=True, verbose=False,
        quiet=False, paramfile=None, in_tmpdir=True,
        **kwargs):

        import shutil
        from subprocess import PIPE, run
        from glob import glob

        old_pwd = os.getcwd()

        try:
            basename = os.path.basename(filename)
            dirname  = os.path.dirname(filename)

            if tmpdir is None:
                tmpdir = f"tmpsnid_{self._snidid}"

            if not os.path.isdir(tmpdir):
                os.makedirs(tmpdir, exist_ok=True)

            if in_tmpdir:
                os.chdir(tmpdir)
                self._tmpfile = f"snid_{self._snidid}_spectofit.ascii"
            else:
                self._tmpfile = os.path.join(tmpdir, f"snid_{self._snidid}_spectofit.ascii")

            shutil.copy(filename, self._tmpfile)

            tmpbase = os.path.basename(self._tmpfile).split(".")[0]
            snid_cmd = self.build_snid_command(
                self._tmpfile, param=paramfile, verbose=verbose, **kwargs
            )

            self._result = run(
                snid_cmd.split(),
                stdout=PIPE,
                stderr=PIPE,
                universal_newlines=True
            )

            if verbose:
                print(f" running: {snid_cmd}")
                print(self._result.stdout.split("\n"))

            if self._result.returncode != 0:
                warnings.warn("SNID returncode is not 0, suggesting an error")
                return None

            if "correlation function is all zero!" in self._result.stdout:
                warnings.warn("SNID failed: correlation function is all zero")
                return None

            if "PEAKFIT: fit quits before half peak points!" in self._result.stdout:
                warnings.warn("SNID failed: PEAKFIT quits before half peak points")
                return None

            datafile = f"{tmpbase}_snidflux.dat"
            modelfiles = glob(f"{tmpbase}_comp*_snidflux.dat")
            snidout = f"{tmpbase}_snid.output"

            try:
                result = SNIDReader._read_snidoutput_(snidout)
            except FileNotFoundError:
                print(" SNID RETURN CODE ".center(40, "-"))
                print(self._result.stdout)
                print("".center(40, "-"))
                return None

            data = SNIDReader._read_snidflux_(datafile)
            models = pandas.concat({
                int(f_.split("comp")[-1].split("_")[0]):
                SNIDReader._read_snidflux_(f_)
                for f_ in modelfiles
            })

            if fileout is None:
                if dirout is None:
                    dirout = dirname

                basename_noext, ext = os.path.splitext(basename)
                fileout = os.path.join(dirout, basename_noext + "_snid.h5")

            elif not fileout.endswith("h5"):
                fileout += ".h5"

            result.to_hdf(fileout, key="results", format="table")
            data.to_hdf(fileout, key="data", format="table")
            models.to_hdf(fileout, key="models", format="table")

            if not quiet:
                print(f"snid run was successful: data stored at {fileout}")

            return fileout

        finally:
            if cleanout:
                self._cleanup_run_(tmpdir, old_pwd=old_pwd)
            elif old_pwd is not None:
                os.chdir(old_pwd)

    def _cleanup_run_(self, tmpdir, old_pwd=None):
        """Clean SNID temporary files safely."""

        # First, always go back to the original directory
        if old_pwd is not None:
            try:
                os.chdir(old_pwd)
            except Exception as e:
                warnings.warn(f"Could not return to original directory {old_pwd}: {e}")

        # Then remove the whole temporary directory
        if tmpdir is not None and os.path.isdir(tmpdir):
            try:
                shutil.rmtree(tmpdir)
            except Exception as e:
                warnings.warn(f"Could not remove temporary directory {tmpdir}: {e}")

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
