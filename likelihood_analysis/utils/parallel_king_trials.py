import time
import numpy as np
import multiprocessing as mp
import csky as cy


def _fit_trials_worker(
    tr: cy.trial.TrialRunner,
    n_trials,
    seeds,
    n_sig=0,
    poisson=False,
    queue=None,
    i_job=0,
    progress=None,
    **fitter_args,
):
    fits = []

    try:
        for i in range(n_trials):

            if progress is not None and progress[i_job] < 0:
                break

            trial = tr.get_one_trial(
                n_sig=n_sig,
                poisson=poisson,
                seed=seeds[i],
            )

            fit = tr.get_one_fit_from_trial(
                trial,
                **fitter_args,
            )

            fit = list(fit)
            fit.append(seeds[i])

            fits.append(fit)

            if progress is not None:
                progress[i_job] = i + 1

    except KeyboardInterrupt:
        if progress is not None:
            progress[i_job] = -1

    fits = np.asarray(fits)

    if queue is None:
        return fits

    queue.put((i_job, fits))


def get_many_fits_from_trials(
    tr: cy.trial.TrialRunner,
    n_trials,
    n_sig=0,
    poisson=False,
    seed=None,
    mp_cpus=None,
    logging=True,
    **fitter_args,
):

    if mp_cpus is None:
        mp_cpus = tr.mp_cpus

    n_trials = int(n_trials)
    mp_cpus = min(n_trials, mp_cpus)

    random = cy.utils.get_random(seed)
    seeds = random.randint(2**32, size=n_trials)

    if logging:
        print(
            f"Performing {n_trials} trials with n_sig={n_sig} "
            f"using {mp_cpus} core{'s' if mp_cpus != 1 else ''}:"
        )

    if mp_cpus == 1:

        fits = _fit_trials_worker(
            tr,
            n_trials=n_trials,
            seeds=seeds,
            n_sig=n_sig,
            poisson=poisson,
            **fitter_args,
        )

        return tr._fits_np_to_Arrays(fits)

    split = np.array_split(np.arange(n_trials), mp_cpus)

    n_jobs = len(split)

    queue = mp.Queue()

    progress = mp.Array(
        "i",
        np.zeros(n_jobs, dtype=int),
    )

    procs = []

    for i_job in range(n_jobs):

        idx = split[i_job]

        proc = mp.Process(
            target=_fit_trials_worker,
            kwargs=dict(
                tr=tr,
                n_trials=len(idx),
                seeds=seeds[idx],
                n_sig=n_sig,
                poisson=poisson,
                queue=queue,
                i_job=i_job,
                progress=progress,
                **fitter_args,
            ),
        )

        procs.append(proc)

        proc.start()

    try:

        n_complete = 0

        while n_complete < n_trials:

            n_complete = np.sum(progress)

            if logging:
                print(
                    f"\r{n_complete}/{n_trials}",
                    end="",
                    flush=True,
                )

            time.sleep(1 if logging else 0.25)

    except KeyboardInterrupt:

        for i_job in range(n_jobs):
            progress[i_job] = -2

        print("\nKeyboardInterrupt: terminating workers.")

    fitss = [queue.get() for _ in procs]

    for proc in procs:
        proc.join()

    fitss = [fits for _, fits in sorted(fitss)]

    fits = np.concatenate(fitss)

    if logging:
        print(f"\r{len(fits)}/{n_trials} done")

    return tr._fits_np_to_Arrays(fits)