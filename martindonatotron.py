# -*- coding: utf-8 -*-
import argparse
import json
import matplotlib.pyplot as pyplot
from matplotlib.patches import Patch
from itertools import cycle
from xlrd import open_workbook

from translation import translate, get_names
from structures import Municipality, World, generate_gradient
from provinces import PROVINCES_TO_IMPORT


# https://gis.stackexchange.com/questions/131716/plot-shapefile-with-matplotlib
RAW_COLOURS = ('b', 'g', 'r', 'c', 'm', 'y')
# Palettes
RAW_COLOURS_1 = ["#669c55", "#9749b9", "#ba8040", "#757cad", "#b54958"]
RAW_COLOURS_2 = ["#b65090", "#56ae6c", "#7065bb", "#ad993c", "#ba4f45"]
RAW_COLOURS_3 = ["#8c5db3", "#65ac6b", "#c3515a", "#c29846", "#5d5132"]
RAW_COLOURS = RAW_COLOURS_1 + RAW_COLOURS_2 + RAW_COLOURS_3
RAW_COLOURS = [
    # light blue
    "#6d8dd7",
    # dark red
    "#8d2c66",
    # light gold
    "#d29b3a",
    # light green
    "#60a756",
    # gold
    "#998039",
    # purple
    "#6971d7",
    # green
    "#9ab03e",
    # dark purple
    "#5c3889",
    # pink
    "#ca74c7",
    # teal
    "#45c097",
    # orange
    "#c76334",
    # dark pink
    "#d76092",
    # brown
    "#ae4837",
    # weird
    "#be4a5b",
] + RAW_COLOURS * 5
COLOURS = cycle(RAW_COLOURS[:3])


def import_population(verbose):
    print('========== IMPORTING POPULATION SIZES =========')
    # downloaded from http://www.ine.es/dynt3/inebase/index.htm?padre=525
    wb = open_workbook(filename='pobmun/pobmun16.xlsx')

    # Only one sheet
    sheet = wb.sheet_by_index(0)
    population = {}
    all_provinces = set()
    read_provinces = set()
    for nrow in xrange(sheet.nrows):
        row = sheet.row(nrow)
        # Skip headers
        try:
            int(row[0].value)
        except:
            continue

        province = translate(row[1].value)
        name = translate(row[3].value, province)

        if verbose:
            if province not in all_provinces:
                all_provinces.add(province)
                print(u'province population: {}'.format(province))

        if province not in PROVINCES_TO_IMPORT:
            continue

        read_provinces.add(province)

        pop = row[4].value
        population[name] = int(pop)

    # Add special regions (Facerías in Navarra)
    SPECIAL_REGIONS = (
        u'FACERÍA 2', u'Sin nombre', u'FACERÍA 8', u'FACERÍA 9',
        u'FACERÍA 10', u'FACERÍA 11', u'FACERÍA 14',
        u'FACERÍA 15', u'FACERÍA 16', u'FACERÍA 17',
        u'FACERÍA 18 "REMENDÍA"', u'FACERÍA 21', u'FACERÍA 22',
        u'MONTE COMÚN DE LAS AMÉSCOAS', u'FACERÍA 24 "LARRAIZA',
        u'FACERÍA 26', u'FACERÍA 27', u'FACERÍA 28',
        u'FACERÍA 29 "ARAMBELTZ"', u'FACERÍA 30 "SAMINDIETA"',
        u'FACERÍA 31', u'FACERÍA 32', u'FACERÍA 35',
        u'FACERÍA 36', u'FACERÍA 37', u'FACERÍA 38',
        u'FACERÍA 39', u'FACERÍA 40', u'FACERÍA 41',
        u'FACERÍA 42', u'FACERÍA 43', u'FACERÍA 44',
        u'FACERÍA 45', u'FACERÍA 46', u'FACERÍA 49',
        u'FACERÍA 50', u'FACERÍA 52', u'FACERÍA 53',
        u'FACERÍA 55', u'FACERÍA 56', u'FACERÍA 62',
        u'FACERÍA 63', u'FACERÍA 65', u'FACERÍA 67',
        u'FACERÍA 70', u'FACERÍA 71', u'FACERÍA 74',
        u'FACERÍA 75', u'FACERÍA 76', u'FACERÍA 79',
        u'FACERÍA 81', u'FACERÍA 82', u'FACERÍA 83',
        u'FACERÍA 84', u'FACERÍA 85', u'FACERÍA 86',
        u'FACERÍA 87', u'FACERÍA 88', u'FACERÍA 91',
        u'FACERÍA 92', u'FACERÍA 103', u'FACERÍA 104',
        u'FACERÍA 105', u'FACERÍA 106', u'FACERÍA 107',
        u'FACERÍA 108', u'BARDENAS REALES', u'SIERRA DE ANDÍA',
        u'SIERRA DE ARALAR', u'SIERRA DE LÓQUIZ',
        u'SIERRA DE URBASA',

        # Oza dos Ríos and Cesuras got merged. Cesuras will
        # be taking all population. Both will be printed
        u'Oza dos Ríos',
    )
    for name in SPECIAL_REGIONS:
        population[name] = 0

    # Return some stats
    num_muns = len(population)
    num_provinces = len(read_provinces)
    tmpl = 'Read population of {muns} municipalities in {provs} provinces'
    print(tmpl.format(muns=num_muns, provs=num_provinces))
    if verbose:
        print('Provinces read: {}'.format(read_provinces))

    return population


def decode(to_decode):
    return unicode(to_decode, 'cp1252')


def check_in_population(verbose, name, population):
    if name not in population:
        # Try to find a match
        letters = {n for n in name if n not in ['(', ')']}
        matchs = [(match, len({n for n in match} & letters))
                  for match in population]
        matchs.sort(key=lambda x: x[1], reverse=True)
        if verbose:
            print(u'Name not found: {}'.format(name))
        candidates = [c for c, _ in matchs[:3]]
        if verbose:
            print(u'Candidates: {}'.format(candidates))
        print('{}: {},'.format(repr(name), repr(candidates[0])))
        best_candidate = candidates[0]
        population[name] = best_candidate

    assert name in population


GIS_FILE = "Municipios_ETRS89_30N/Municipios_ETRS89_30N"


def import_map(verbose, population):
    # Dalaying the import of fiona to avoid (weird) issues with shapely library
    import fiona
    print('========== IMPORTING MAPS =========')
    # downloaded from
    # https://www.arcgis.com/home/item.html?id=2e47bb12686d4b4b9d4c179c75d4eb78#!

    world = World(RAW_COLOURS, verbose=verbose)

    all_provinces = set()
    read_provinces = set()

    print('========== IMPORTING IN FIONA FORMAT =========')
    geometries = {}
    with fiona.open(GIS_FILE + '.shp') as fd:
        for data in fd:

            raw_name = data['properties']['Texto']
            # SKIPPING_MUN = (u'FACERÍA', 'SIERRA', 'BARDENAS',
            #                 u'MONTE COMÚN', 'Sin nombre')
            SKIPPING_MUN = []
            skipping = any(skip in raw_name for skip in SKIPPING_MUN)
            if skipping:
                if verbose:
                    print(u'Skipping {}'.format(raw_name))
                continue

            name, province_name = get_names(raw_name,
                                            data['properties']['Provincia'])

            if province_name not in all_provinces:
                all_provinces.add(province_name)

                if verbose:
                    print(u'Province map: {}'.format(province_name))

            if province_name not in PROVINCES_TO_IMPORT:
                continue

            read_provinces.add(province_name)

            if verbose:
                print(u'Importing {}'.format(raw_name))

            check_in_population(verbose, name, population)
            if name in geometries:
                print u'DUPLICATED NAME {} {}'.format(name, province_name)

            geometries[name] = data

    for name, data in geometries.items():

        mun = Municipality(name, population[name],
                           geometry=data['geometry'])
        world.append(mun)

    world.geometries = geometries

    if verbose:
        print('all_provinces', all_provinces)

    num_provs = len(read_provinces)
    num_maps = len(world.municipalities)
    print('Imported {} maps on {} provinces'.format(num_maps, num_provs))

    if len(world.geometries) != len(world.municipalities):
        geom = sorted(world.geometries.keys())
        munic = sorted(m.name for m in world.municipalities)
        for e1, e2 in zip(geom, munic):
            print 'e1', e1, 'e2', e2, '<----' if e1 != e2 else ''
        raise Exception

    print('========== COMPOSING GEOMETRY =========')
    world.prepare(verbose=True)
    print('========== GEOMETRY READY =========')

    return world


def general_map(plot, world):
    for mun in world.municipalities:
        if plot:
            mun.plot()


def equipopulous(plot, number_of_regions, interactive, world,
                 seeds=None):
    print('========== GENERATING EQUIPOPULOUS REGIONS =========')
    world.make_regions(number_of_regions, RAW_COLOURS, interactive, seeds)
    world.print_regions()
    if plot:
        # Speeds up not plotting if not needed
        world.plot_regions()


def histogram(plot, verbose, world):
    NUM_GROUPS = 6
    histogram = world.population_histogram(NUM_GROUPS)
    colours = {}
    available_colours = generate_gradient(NUM_GROUPS)
    # available_colours = ['#ff0000', '#ff4d4d', '#ffe6e6']
    l_colour = []
    l_text = []

    for index, group in enumerate(histogram):
        colour = available_colours[index]
        if group['number']:
            l_colour.append(Patch(color=colour))
            l_text.append('{0} to {1}'.format(group['min_population'],
                                            group['max_population']))

        for mun in group['municipalities']:
            colours[mun.name] = colour

    for mun in world.municipalities:

        colour = colours[mun.name]

        if plot:
            mun.plot(colour, line=verbose)

    if plot:
        pyplot.legend(l_colour, l_text)

def to_numbers(number):
    if number / 1000000:
        return '{}M'.format(number / 1000000)

    if number / 1000:
        return '{}K'.format(number / 1000)

    return '{}'.format(number)

def only_map(plot, verbose, world, pop_max, pop_min):

    dark_colour = "#A9C5EB"
    clear_colour = "#EAF1FB"
    l_colour = [
        Patch(color=dark_colour),
        #Patch(color=clear_colour),
    ]


    l_text = [
        '{0} to {1}'.format(to_numbers(pop_min), to_numbers(pop_max)),
        #'Everything else',
    ]


    for mun in world.municipalities:

        if pop_max >= mun.population >= pop_min:
            mun.plot(dark_colour, line=verbose)
        else:
            mun.plot(clear_colour, line=verbose)

    if plot:
        pyplot.legend(l_colour, l_text)


def tests(world, verbose):
    print('Checking stuff')

    # cartagena = world.by_name['Cartagena']
    # mazarron = world.by_name[u'Mazarrón']
    # murcia = world.by_name[u'Murcia']
    # lorca = world.by_name[u'Lorca']
    # alcantarilla = world.by_name['Alcantarilla']

    # Check world bounds
    assert not [m for m in world.municipalities
                if world.bounds.left > m.bounds.left]
    assert not [m for m in world.municipalities
                if world.bounds.top < m.bounds.top]
    assert not [m for m in world.municipalities
                if world.bounds.right < m.bounds.right]
    assert not [m for m in world.municipalities
                if world.bounds.bottom > m.bounds.bottom]
    cartagena_cont = [u'Torre-Pacheco', u'Alc\xe1zares, Los', u'Cartagena',
                      u'Murcia', u'Mazarr\xf3n', u'Uni\xf3n, La',
                      u'Fuente \xc1lamo de Murcia']
    # assert world.contiguous['Cartagena'] == set(cartagena_cont), world.contiguous['Cartagena']

    dumped = json.dumps(world.dump())
    new_world = World.load(json.loads(dumped), verbose=verbose)

    assert world == world
    assert new_world == new_world

    # assert world == new_world

    # import pdb; pdb.set_trace()
    return new_world


def main():
    parser = argparse.ArgumentParser(description='Create maps')
    parser.add_argument('--import-population-only', dest='population_only',
                        action='store_true',
                        help='Exit after importing population from files. '
                             'This is aimed at debug')
    parser.add_argument('--import-map-only', dest='map_only',
                        action='store_true',
                        help='Exit after importing map from files. '
                             'This is aimed at debug')
    parser.add_argument('-t', dest='type', type=str, default='regions',
                        help='Type of map: regions, histogram, only, general, tests')
    parser.add_argument('-p', dest='plot', action='store_true',
                        help='Print result')
    parser.add_argument('-n', dest='number', type=int, default=5,
                        help='Number of elements (regions, etc)')
    parser.add_argument('-i', dest='interactive', action='store_true',
                        help='Enter interactive mode')
    parser.add_argument('--seed', dest='seed', type=str, default=None,
                        help='Seed the regions and municipalities')
    parser.add_argument('-v', dest='verbose', action='store_true',
                        help='Enable verbose ooutput')
    parser.add_argument('--generate', dest='out_file', type=str, default=None,
                        help='file to generate world output')
    parser.add_argument('--in', dest='in_file', type=str, default=None,
                        help='file to import world definition')
    parser.add_argument('--only-max', dest='only_max', type=int, default=1000000,
                        help='Upper limit of population to print on an only map')
    parser.add_argument('--only-min', dest='only_min', type=int, default=0,
                        help='Lower limit of population to print on an only map')

    args = parser.parse_args()

    if not args.in_file:
        population = import_population(args.verbose)
        if args.population_only:
            print('Exiting after importing the populations')
            exit(0)

        world = import_map(args.verbose, population)
        if args.map_only:
            print('Exiting after importing the maps')
            exit(0)

    if args.in_file:
        print('========== IMPORTING FROM FILE  =======')
        with open(args.in_file) as fp:
            world = World.load(json.load(fp), args.verbose)

    if args.out_file:
        with open(args.out_file, 'w') as fp:
            json.dump(world.dump(), fp)

        exit(0)

    # tests(world, args.verbose)

    pyplot.figure()
    pyplot.axis('equal')

    if args.type == 'general':
        general_map(args.plot, world)
    elif args.type == 'histogram':
        histogram(args.plot, args.verbose, world)

    elif args.type == 'regions':
        seeds = []
        number = args.number
        if args.seed:
            # Generate number and seeds
            with open(args.seed) as fp:
                seeds = [[l.strip().decode('utf8') for l in line.split('#')]
                         for line in fp]
            number = len(seeds)
            print('Using seeds', seeds)

        equipopulous(args.plot, number, args.interactive, world, seeds)
    elif args.type == 'only':
        only_map(args.plot, args.verbose, world, args.only_max, args.only_min)
    else:
        raise Exception('Unknown type of map')

    if args.plot:
        pyplot.show()


if __name__ == '__main__':
    main()
