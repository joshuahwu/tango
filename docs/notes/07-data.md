# Data

Where should data files live? What's the size limits?
- size limit for inclusion in repo: 1 MB per file
- all in `data/` folder under root
- for any file larger than 1MB, create an [OSF](https://osf.io/dashboard) project (one project per repo) and upload any files there. Can mint a DOI for the project, allows versioning of files. Can write a wget call or python function to download (example from plenoptic):
```
def osf_download(filename):
    r"""Download file from plenoptic OSF page.

    From the OSF project at https://osf.io/ts37w/.

    Downloads the specified file to `plenoptic/data`, extracts and deletes the
    the .tar.gz file (if applicable), and returns the path.

    Parameters
    ----------
    filename : {'plenoptic-test-files.tar.gz', 'ssim_images.tar.gz',
                'ssim_analysis.mat', 'msssim_images.tar.gz',
                'MAD_results.tar.gz',
                'portilla_simoncelli_images.tar.gz',
                'portilla_simoncelli_matlab_test_vectors.tar.gz',
                'portilla_simoncelli_test_vectors.tar.gz',
                'portilla_simoncelli_synthesize.npz',
                'portilla_simoncelli_synthesize_torch_v1.12.0.npz',
                'portilla_simoncelli_synthesize_gpu.npz'}
        Which file to download.

    Returns
    -------
    path : str
        The path to the downloaded directory or file.

    """
    path = op.join(op.dirname(op.realpath(__file__)), '..', 'data', filename)
    if not op.exists(path.replace('.tar.gz', '')):
        print(f"{filename} not found, downloading now...")
        # Streaming, so we can iterate over the response.
        r = requests.get(f"https://osf.io/{OSF_URL[filename]}/download",
                         stream=True)

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024*1024
        wrote = 0
        with open(path, 'wb') as f:
            for data in tqdm.tqdm(r.iter_content(block_size), unit='MB',
                                  unit_scale=True,
                                  total=math.ceil(total_size//block_size)):
                wrote += len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            raise Exception(f"Error downloading {filename}!")
        if filename.endswith('.tar.gz'):
            with tarfile.open(path) as f:
                f.extractall(op.dirname(path))
            os.remove(path)
        print("DONE")
    return path.replace('.tar.gz', '')
```

- any file larger than 5GB does not belong in docs or tests, but potentially useful for workshops. for that, talk with SCC about setting up storage.
