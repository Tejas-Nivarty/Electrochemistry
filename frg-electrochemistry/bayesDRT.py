import numpy as np
import time
import matplotlib.pyplot as plt
from ReadDataFiles import colorFader, convertToImpedanceAnalysis
import pandas as pd
import json
import hashlib
from pathlib import Path
from pandas.util import hash_pandas_object
from tempfile import NamedTemporaryFile

from bayes_drt2.inversion import Inverter

def getDRT(eisData: pd.DataFrame):
    """Gets single DRT of both HMC and MAP methods. From bayesDRT2 default functions.

    Args:
        eisData (pd.DataFrame): DataFrame of EIS data.
        
    Returns:
        tuple(matplotlib.figure.Figure,matplotlib.axes._axes.Axes): fig and ax for further customization if necessary
    """
    freq, Z = convertToImpedanceAnalysis(eisData)
    
    "Fit the data"
    # By default, the Inverter class is configured to fit the DRT (rather than the DDT)
    # Create separate Inverter instances for HMC and MAP fits
    # Set the basis frequencies equal to the measurement frequencies 
    # (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
    inv_hmc = Inverter(basis_freq=freq)
    inv_map = Inverter(basis_freq=freq)

    # Perform HMC fit
    start = time.time()
    inv_hmc.fit(freq, Z, mode='sample',nonneg=True,outliers='auto')
    elapsed = time.time() - start
    print('HMC fit time {:.1f} s'.format(elapsed))

    # Perform MAP fit
    start = time.time()
    inv_map.fit(freq, Z, mode='optimize',nonneg=True,outliers='auto')  # initialize from ridge solution
    elapsed = time.time() - start
    print('MAP fit time {:.1f} s'.format(elapsed))
    
    "Visualize DRT and impedance fit"
    # plot impedance fit and recovered DRT
    fig,axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # plot fits of impedance data
    inv_hmc.plot_fit(axes=axes[0], plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
    inv_map.plot_fit(axes=axes[0], plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

    # add Dirac delta function for RC element
    axes[1].plot([np.exp(-2),np.exp(-2)],[0,10],ls='--',lw=1)

    # Plot recovered DRT at given tau values
    inv_hmc.plot_distribution(ax=axes[1], color='k', label='HMC mean', ci_label='HMC 95% CI')
    inv_map.plot_distribution(ax=axes[1], color='r', label='MAP')

    # axes[1].set_ylim(0,3.5)
    # axes[1].legend()


    fig.tight_layout()
    plt.show()
    
    # "Visualize the recovered error structure"
    # # For visual clarity, only MAP results are shown.
    # # HMC results can be obtained in the same way
    # fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)

    # # plot residuals and estimated error structure
    # inv_map.plot_residuals(axes=axes)

    # # plot true error structure in miliohms
    # p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
    # axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
    # axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
    # axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

    # axes[1].legend()

    # fig.tight_layout()
        
    return (fig, axes)

# def plotManyDRTs(eisDatas: list[pd.DataFrame], title: str, legendList: list[str] = None, logResistance = False, regenerate = False):
#     """Plots many DRTs.

#     Args:
#         eisDatas (list[pd.DataFrame]): List of EIS DataFrames.
#         title (str): Title of DRT plot.
#         legendList (list[str], optional): List of legend elements. Defaults to potential that EIS was taken at.
#         logResistance (bool, optional): Plots resistance in log format for better visualizing series.
#         regenerate (bool, default False): if true, will regenerate DRTs instead of using cached versions. if no cached version, will just use default

#     Returns:
#         tuple(matplotlib.figure.Figure,matplotlib.axes._axes.Axes): fig and ax for further customization if necessary
#     """
#     #plotManyNyquists(eisDatas,title,freqRange,legendList=legendList)
    
#     #automatically finds minimum and maximum of freqRange
#     minFreq = np.inf
#     maxFreq = -np.inf
#     for eisData in eisDatas:
#         currMinFreq = eisData['freq/Hz'].min()
#         currMaxFreq = eisData['freq/Hz'].max()
#         if currMinFreq < minFreq:
#             minFreq = currMinFreq
#         if currMaxFreq > maxFreq:
#             maxFreq = currMaxFreq
#     freqRange = [minFreq,maxFreq]
    
#     freqRange = np.nan_to_num(freqRange,neginf=0.1,posinf=7e6)
#     logFreqRange = -np.log10(freqRange)
#     confInt = 95
    
#     numberOfPlots = len(eisDatas)
#     fig, ax = plt.subplots()
#     ax.set_title(title)
#     timeConstantArray = np.logspace(logFreqRange[0],logFreqRange[1], 1000)
    
#     for i in range(0,numberOfPlots):
        
#         color = colorFader('blue','red',i,numberOfPlots)
        
#         #looks for cached .drt file
#         filename = eisDatas[i].attrs['filename']
#         if 
#         freq, Z = convertToImpedanceAnalysis(eisDatas[i])
#         inv_hmc = Inverter(basis_freq=freq)
#         inv_hmc.fit(freq, Z, mode='sample',nonneg=True,outliers=True)
        
#         #use predict distribution, can adjust percentile to get error bars
#         gammaMean = inv_hmc.predict_distribution(tau=timeConstantArray)
#         gammaLo = inv_hmc.predict_distribution(tau=timeConstantArray,percentile=50-(confInt/2))
#         gammaHi = inv_hmc.predict_distribution(tau=timeConstantArray,percentile=50+(confInt/2))
        
#         if legendList == None:
#             #finds potential at which EIS was taken
#             potential = eisDatas[i]['<Ewe>/V'].mean()*1000
#             ax.plot(timeConstantArray,gammaMean,color=color,label='{:3.0f}'.format(potential)+r' $mV_{ref}$')
#         else:
#             ax.plot(timeConstantArray,gammaMean,color=color,label=legendList[i])
            
#         ax.fill_between(timeConstantArray,gammaLo,gammaHi,color=color,alpha=0.2,label='_')
        
#         continue
    
#     ax.legend()
#     ax.set(ylabel = r'$\gamma$ ($\Omega$)',
#             xlabel = r'Time Constant (s)',
#             xscale='log')
    
#     if logResistance:
#         ax.set_yscale('symlog', linthresh=1)
#         ax.set_ylim([0,ax.get_ylim()[1]])
#     plt.tight_layout()
#     plt.show()
    
#     return (fig, ax)

def _norm_df_hash(df):
    """Hash content deterministically: sort columns; include index."""
    try:
        cols = sorted(df.columns)
        dfv = df[cols]
    except Exception:
        dfv = df  # if weird columns, just use as-is
    try:
        hbytes = hash_pandas_object(dfv, index=True).values.tobytes()
        return hashlib.blake2b(hbytes, digest_size=20).hexdigest()
    except Exception:
        return None

def _file_sig(path_str: str):
    """Cheap signature of the source file (size + mtime_ns)."""
    try:
        p = Path(path_str)
        st = p.stat()
        return f"{st.st_size}-{st.st_mtime_ns}"
    except Exception:
        return None

def _save_npz_exact_atomically(path: Path, **arrays):
    """Write an npz to an exact filename (no .npz appended), atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = NamedTemporaryFile(delete=False, dir=path.parent)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **arrays)
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

def _load_npz_exact(path: Path):
    """Load an npz from an exact filename; fall back to .npz if needed."""
    if path.exists():
        return np.load(path, allow_pickle=False)
    alt = Path(str(path) + ".npz")
    if alt.exists():
        return np.load(alt, allow_pickle=False)
    raise FileNotFoundError(str(path))

def plotManyDRTs(eisDatas: list[pd.DataFrame],
                 title: str,
                 legendList: list[str] = None,
                 logResistance: bool = False,
                 regenerate: bool = False,
                 debug: bool = False):
    """
    Plots many DRTs. Caches per-input EIS file as <same path>.drt (npz payload).
    Cache identity = (normalized DataFrame hash, solver_opts). Fallback = (file size, mtime).
    Set regenerate=True to force recompute. Set debug=True to see cache decisions.

    Returns:
        (fig, ax)
    """
    # ---- global freq range -> plotting τ grid -------------------------------
    minFreq, maxFreq = np.inf, -np.inf
    for df in eisDatas:
        fmin = df['freq/Hz'].min()
        fmax = df['freq/Hz'].max()
        if fmin < minFreq: minFreq = fmin
        if fmax > maxFreq: maxFreq = fmax
    freqRange = [minFreq, maxFreq]
    freqRange = np.nan_to_num(freqRange, neginf=0.1, posinf=7e6)
    logFreqRange = -np.log10(freqRange)  # τ = 1/f, in log space
    confInt = 95
    tau_plot = np.logspace(logFreqRange[0], logFreqRange[1], 1000)

    fig, ax = plt.subplots()
    ax.set_title(title)
    n = len(eisDatas)

    # options that define cache identity (tweak if you change solver behavior)
    solver_opts = dict(mode="sample", nonneg=True, outliers=True, confInt=confInt)

    for i, eis in enumerate(eisDatas):
        color = colorFader('blue', 'red', i, n)  # your helper

        # derive cache path: C:\...\foo.mpt -> C:\...\foo.drt
        filename = eis.attrs.get('filename', f"unnamed_{i}.mpt")
        try:
            drt_path = Path(filename).with_suffix(".drt")
        except Exception:
            drt_path = Path(f"./unnamed_{i}.drt")

        # signatures
        df_hash  = _norm_df_hash(eis)
        f_sig    = _file_sig(filename)

        # try cache ------------------------------------------------------------
        tau_src = gammaMean = gammaLo = gammaHi = None
        use_cache = False
        miss_reason = "unknown"

        if not regenerate:
            try:
                npz = _load_npz_exact(drt_path)
                try:
                    tau_src   = npz["tau"]
                    gammaMean = npz["gamma"]
                    gammaLo   = npz["gamma_lo"] if "gamma_lo" in npz.files else None
                    gammaHi   = npz["gamma_hi"] if "gamma_hi" in npz.files else None
                    meta_raw  = npz["meta"][0]
                finally:
                    npz.close()

                # tolerant decode for NumPy 2.0 (np.str_) and older (bytes)
                if isinstance(meta_raw, (bytes, np.bytes_)):
                    meta = json.loads(meta_raw.decode("utf-8"))
                else:
                    meta = json.loads(str(meta_raw))

                meta_df  = meta.get("data_hash")
                meta_opt = meta.get("solver_opts")
                meta_sig = meta.get("file_sig")

                # primary identity
                if df_hash is not None and meta_df == df_hash and meta_opt == solver_opts:
                    use_cache = True
                    miss_reason = "hit(df_hash)"
                # fallback if df hash is None or changed but source file is identical
                elif meta_opt == solver_opts and meta_sig is not None and meta_sig == f_sig:
                    use_cache = True
                    miss_reason = "hit(file_sig)"
                else:
                    if meta_opt != solver_opts:
                        miss_reason = f"opts_mismatch"
                    elif (df_hash is None) or (meta_df != df_hash):
                        miss_reason = "hash_mismatch"
            except FileNotFoundError:
                miss_reason = "no_cache_file"
            except Exception as e:
                miss_reason = f"load_error:{e!r}"

        if debug:
            status = "HIT" if use_cache else "MISS"
            print(f"[DRT cache:{status}] {drt_path} :: {miss_reason}")

        if not use_cache:
            # compute fresh ----------------------------------------------------
            freq, Z = convertToImpedanceAnalysis(eis)
            inv_hmc = Inverter(basis_freq=freq)
            inv_hmc.fit(freq, Z, mode='sample', nonneg=True, outliers=True)

            # file-specific τ grid based on this file's freq span
            fmin = float(np.nanmin(freq)) if np.isfinite(np.nanmin(freq)) else 1e-3
            fmax = float(np.nanmax(freq)) if np.isfinite(np.nanmax(freq)) else 1e6
            fmin = max(fmin, 1e-12); fmax = max(fmax, 1e-12)
            log_tau_min = -np.log10(max(fmax, 1e-12))
            log_tau_max = -np.log10(max(fmin, 1e-12))
            tau_src = np.logspace(log_tau_min, log_tau_max, 1500)

            gammaMean = inv_hmc.predict_distribution(tau=tau_src)
            gammaLo   = inv_hmc.predict_distribution(tau=tau_src, percentile=50 - (confInt/2))
            gammaHi   = inv_hmc.predict_distribution(tau=tau_src, percentile=50 + (confInt/2))

            # save cache (exact filename) with np.str_ for NumPy 2.x
            try:
                meta = {
                    "source": str(filename),
                    "data_hash": df_hash,
                    "solver_opts": solver_opts,
                    "file_sig": f_sig,
                    "schema": 1
                }
                _save_npz_exact_atomically(
                    drt_path,
                    tau=tau_src,
                    gamma=gammaMean,
                    gamma_lo=gammaLo,
                    gamma_hi=gammaHi,
                    meta=np.array([json.dumps(meta)], dtype=np.str_)
                )
            except Exception as e:
                print(f"[DRT cache] write failed for {drt_path}: {e}")

        # interpolate to common plotting τ grid (log space)
        xi = np.log10(tau_plot)
        x  = np.log10(tau_src)
        g   = np.interp(xi, x, gammaMean, left=np.nan, right=np.nan)
        glo = np.interp(xi, x, gammaLo,   left=np.nan, right=np.nan) if gammaLo is not None else None
        ghi = np.interp(xi, x, gammaHi,   left=np.nan, right=np.nan) if gammaHi is not None else None

        # label
        if legendList is None:
            potential = eis['<Ewe>/V'].mean() * 1000
            label = f"{potential:3.0f} mV$_{{ref}}$"
        else:
            label = legendList[i]

        ax.plot(tau_plot, g, color=color, label=label)
        if glo is not None and ghi is not None:
            ax.fill_between(tau_plot, glo, ghi, color=color, alpha=0.2, label='_')

    ax.legend()
    ax.set(ylabel=r'$\gamma$ ($\Omega$)',
           xlabel=r'Time Constant (s)',
           xscale='log')

    if logResistance:
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_ylim([0, ax.get_ylim()[1]])

    plt.tight_layout()
    plt.show()
    return (fig, ax)