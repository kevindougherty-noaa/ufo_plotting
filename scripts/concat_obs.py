import argparse
import xarray as xr

def concat_nc(infiles, outfile):
    ds = xr.open_mfdataset(infiles+'*', concat_dim="nlocs")
    ds.to_netcdf(outfile)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--infiles', help='path to start of name of UFO netCDF diags without wildcard', required=True)
    ap.add_argument('-o', '--outfile', help="concatenated filename", required=True)
    MyArgs = ap.parse_args()
    concat_nc(MyArgs.infiles, MyArgs.outfile)
