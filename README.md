Martindonatotron
===============

Generate different maps of Spain, splitting by population

```
usage: martindonatotron.py [-h] [--import-population-only] [--import-map-only]
                           [-t TYPE] [-p] [-n NUMBER] [-i] [--seed SEED] [-v]
                           [--generate OUT_FILE] [--in IN_FILE]
                           [--only-max ONLY_MAX] [--only-min ONLY_MIN]

Create maps

optional arguments:
  -h, --help            show this help message and exit
  --import-population-only
                        Exit after importing population from files. This is
                        aimed at debug
  --import-map-only     Exit after importing map from files. This is aimed at
                        debug
  -t TYPE               Type of map: regions, histogram, only, general, tests
  -p                    Print result
  -n NUMBER             Number of elements (regions, etc)
  -i                    Enter interactive mode
  --seed SEED           Seed the regions and municipalities
  -v                    Enable verbose ooutput
  --generate OUT_FILE   file to generate world output
  --in IN_FILE          file to import world definition
  --only-max ONLY_MAX   Upper limit of population to print on an only map
  --only-min ONLY_MIN   Lower limit of population to print on an only map
```

In the repo there are pregenerated json files with Murcia (murcia.json) and the whole peninsular spain to play with (peninsula.json)

Remember to add -p to display the map.

# requirements

Install with the requirements.txt file

    pip install -r requirements.txt

# Map types

The following types of map are supported

## regions

Generate NUMBER equipopulous regions. By default, based on the most populated cities. A file can be created to seed the 
initial cities per region. Check example_seed.txt.

e.g. 
```
python2 martindonatotron.py --in murcia.json -t regions -n 5 -p
```

interactive mode can be used to generate the seed as well.


## general

Display all municipalities in different colours

## histogram

Display municipalities with different colours based on their population

## only

Display only the municipalities with population between --only-min and --only-max. The rest will appear in a lighter shade.
