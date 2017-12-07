# -*- coding: utf-8 -*-
from collections import defaultdict, namedtuple
from random import choice, seed
from shapely.geometry import asShape
from shapely.geometry import mapping
import progressbar
from colour import Color

import matplotlib.pyplot as pyplot
from matplotlib.patches import Patch

Box = namedtuple('Box', ['left', 'bottom', 'right', 'top'])

# Initialize to consistent results
seed(0)


def generate_gradient(steps):
    '''
    Generate a colour gradient
    '''
    red = Color("Khaki")
    colors = list(red.range_to(Color("DarkOliveGreen"), steps))
    color_values = reversed([c.get_hex() for c in colors])
    return list(color_values)


def box_check(bbox1, bbox2):
    left1, bottom1, right1, top1 = bbox1
    left2, bottom2, right2, top2 = bbox2

    if left2 > right1:
        return False

    if left1 > right2:
        return False

    if bottom2 > top1:
        return False

    if bottom1 > top2:
        return False

    return True


class Municipality(object):

    def __init__(self, name, population, geometry, colour=None):
        self.name = name
        self.population = population
        self.shapely = asShape(geometry)

        self.colour = colour
        self._bounds = None

    def dump(self):
        return {
            'name': self.name,
            'population': self.population,
            'colour': self.colour,
            'geometry': mapping(self.shapely),
        }

    @staticmethod
    def load(data):
        return Municipality(name=data['name'], population=data['population'],
                            geometry=data['geometry'], colour=data['colour'])

    def grid_coordinate(self, grid):
        left, bottom, right, top = self.bounds
        for index, hstep in enumerate(grid['horizontal']):
            if left < hstep:
                horizontal = index
                break

        for index, vstep in enumerate(grid['vertical']):
            if bottom < vstep:
                vertical = index
                break

        self.grid = (horizontal, vertical)
        return horizontal, vertical

    def __repr__(self):
        TMPL = u'<Mun: {name} ({pop})>'
        to_display = TMPL.format(name=self.name, pop=self.population)
        return to_display.encode('ascii', errors='replace')

    def contiguous(self, mun):
        coincident_points = set(self.coord) & set(mun.coord)
        if len(coincident_points) > 2:
            return True

        return False

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = Box(*self.shapely.bounds)

        return self._bounds

    def shape_contiguous(self, mun):
        # Check bounds
        if not box_check(self.bounds, mun.bounds):
            return False

        # print 'Distance', self, mun, mun.shapely.distance(self.shapely)
        if mun.shapely.distance(self.shapely) == 0.0:
            return True

        return False

    def cross_bbox(self, mun):
        # Check bboxes
        bbox1 = self.shape.bbox
        bbox2 = mun.shape.bbox

        left1, bottom1, right1, top1 = bbox1
        left2, bottom2, right2, top2 = bbox2

        if left2 > right1:
            return False

        if left1 > right2:
            return False

        if bottom2 > top1:
            return False

        if bottom1 > top2:
            return False

        return True

    @property
    def parts(self):
        '''
        Return the different parts of the municipality in
        the format

        [
            ([x1, x2...], [y1, y2...]),
            ([x1, x2...], [y1, y2...]),
            ...
        ]

        '''
        # From shapely

        geo_type = self.shapely.__geo_interface__['type']
        if geo_type == 'Polygon':
            shapely_coordinates = [zip(*self.shapely.shell)]
            return shapely_coordinates

        # Multipolygon
        parts = [coord[0] for coord in
                 self.shapely.__geo_interface__['coordinates']]
        shapely_coordinates = [zip(*part) for part in parts]
        return shapely_coordinates

        # From shape
        parts = self.shape.parts
        num_parts = len(parts)

        slices = [(parts[i], parts[i + 1] if i + 1 < num_parts else -1)
                  for i in xrange(num_parts)]

        lx, ly = zip(*self.shape.points)
        ret = [(lx[start:end], ly[start: end]) for start, end in slices]
        return ret

    def plot(self, colour=None, line=True):
        if colour is None:
            colour = self.colour

        # Plot each of the parts of the municipality
        for x, y in self.parts:
            pyplot.fill(x, y, colour)
            # Show line
            if line:
                pyplot.plot(x, y, 'k:')


class Region(object):

    def __init__(self, world, colour, verbose=False, name=None):
        self.verbose = verbose
        self.municipalities = set()
        self._population = 0
        self._world = world
        self.colour = colour
        self._contiguous = set()
        self._keep_name = name
        self._shape = None

    def __repr__(self):
        TMPL = u'<Region: {name} ({pop})>'
        to_display = TMPL.format(name=self.name, pop=self.population)
        return to_display.encode('ascii', errors='replace')

    @property
    def human_size(self):
        unit = ''
        if self.population / 1000000:
            result = self.population * 1.0 / 1000000
            unit = 'M'
        elif self.population / 1000:
            result = self.population * 1.0 / 1000
            unit = 'K'
        else:
            result = self.population

        return '{size:.1f}{unit}'.format(size=result, unit=unit)

    def print_region(self):
        tmpl = (u'Region "{name}" with {size} municipalities and '
                'population {pop}')
        if self.verbose:
            tmpl += ': {muns}'

        msg = tmpl.format(size=len(self.municipalities), pop=self.population,
                          muns=self.municipalities, name=self.name)
        print(msg)

    @property
    def name(self):
        '''
        Return the name of the municipality with most population
        '''
        if not self.municipalities:
            return 'EMPTY'
        if self._keep_name:
            return self._keep_name

        mun = max(self.municipalities, key=lambda x: x.population)
        return mun.name

    def add(self, municipality, check_contiguous=False, skip_check=False,
            update_shape=False):
        if not skip_check:
            if municipality.name not in self._world.in_no_region:
                return False

        if check_contiguous:
            if municipality not in self.contiguous:
                return False

        # print(u'Adding municipality {} to Region: {}'.format(municipality,
        #                                                      self.name))

        self.municipalities.add(municipality)
        self._population += municipality.population

        self._contiguous.update(self._world.contiguous[municipality.name])
        current = {mun.name for mun in self.municipalities}
        self._contiguous = self._contiguous - current

        if update_shape:
            if not self._shape:
                self._shape = municipality.shapely
            else:
                self._shape = self.shape.union(municipality.shapely)

        if municipality.name in self._world.in_no_region:
            self._world.in_no_region.remove(municipality.name)

        return True

    def remove(self, municipality):
        self.municipalities.remove(municipality)
        self._population -= municipality.population
        self._contiguous.update(self._world.contiguous[municipality.name])
        current = {mun.name for mun in self.municipalities}
        self._contiguous = self._contiguous - current
        self._world.in_no_region.add(municipality.name)

    def available_contiguous(self, in_no_region):
        cnames = {c.name for c in self.contiguous}
        available = cnames & in_no_region
        return available

    def sort_candidates(self, candidates_to_add):
        '''
        Sort municipalities that are candidates to be added by prefered
        '''
        # LERO

    @property
    def shape(self):
        if not self._shape:
            from shapely.ops import cascaded_union
            all_shapes = (m.shapely for m in self.municipalities)
            self._shape = cascaded_union(all_shapes)
        return self._shape

    @property
    def shape_centre(self):
        if self._shape is None:
            return None

        return self._shape.centroid

    @property
    def population(self):
        return self._population

    @property
    def contiguous(self):
        '''
        Return all the contiguous municipalities
        '''
        return {self._world.by_name[mun] for mun in self._contiguous}

        contiguous = set()
        for mun in self.municipalities:
            mun_contiguous = self._world.contiguous[mun.name]
            contiguous.update(mun_contiguous)

        # Remove all municipalities already in the region
        current = {mun.name for mun in self.municipalities}
        final_by_name = contiguous - current
        return {self._world.by_name[mun] for mun in final_by_name}

    @property
    def parts(self):
        '''
        Return the different parts of the municipality in
        the format

        [
            ([x1, x2...], [y1, y2...]),
            ([x1, x2...], [y1, y2...]),
            ...
        ]

        '''
        # From shapely

        geo_type = self.shape.__geo_interface__['type']
        if geo_type == 'Polygon':
            shapely_coordinates = [zip(*self.shape.shell)]
            return shapely_coordinates

        # Multipolygon
        parts = [coord[0] for coord in
                 self.shape.__geo_interface__['coordinates']]
        shape_coordinates = [zip(*part) for part in parts]
        return shape_coordinates

        # From shape
        parts = self.shape.parts
        num_parts = len(parts)

        slices = [(parts[i], parts[i + 1] if i + 1 < num_parts else -1)
                  for i in xrange(num_parts)]

        lx, ly = zip(*self.shape.points)
        ret = [(lx[start:end], ly[start: end]) for start, end in slices]
        return ret

    def plot_big_shape(self, colour=None, line=True):
        if colour is None:
            colour = self.colour

        # Plot each of the parts of the municipality
        for x, y in self.parts:
            pyplot.fill(x, y, colour)
            # Show line
            if line:
                pyplot.plot(x, y, 'k:')

    def plot(self):
        BIG_SHAPE = False
        if BIG_SHAPE:
            return self.plot_big_shape(line=self.verbose)

        USE_GRADIENT = False
        gradient_colours = reversed(generate_gradient(len(self.municipalities)))
        for mun, colour in zip(self.municipalities, gradient_colours):
            if not USE_GRADIENT:
                colour = self.colour
            mun.plot(colour, line=self.verbose)


class World(object):

    def __init__(self, colours, verbose):
        self.municipalities = []
        self.colours = set(colours)
        self.regions = None
        self.in_no_region = []
        self.verbose = verbose
        self.grid = None

    def append(self, mun):
        self.municipalities.append(mun)

    def create_grid(self, bounds):
        NUM_STEPS = 10 if len(self.municipalities) > 35 else 5
        hor_step = (bounds.right - bounds.left) / NUM_STEPS
        ver_step = (bounds.top - bounds.bottom) / NUM_STEPS
        grid = {
            'horizontal': [bounds.left + (i * hor_step)
                           for i in xrange(NUM_STEPS + 1)],
            'vertical': [bounds.bottom + (i * ver_step)
                         for i in xrange(NUM_STEPS + 1)],
        }
        return grid

    def border_region(self, region, mun):
        mun_contiguous = self.contiguous[mun.name]
        shared = {m.name for m in region.contiguous} & mun_contiguous
        return len(shared)

    def dump(self):
        '''
        Generate a big dictionary with all the data to be able to export
        it to file
        '''

        return {
            # 'grid': self.grid,
            'colours': list(self.colours),
            'by_name': {name: m.dump() for name, m in self.by_name.items()},
            # 'by_grid': self.by_grid,
            'contiguous': {name: list(mlist)
                           for name, mlist in self.contiguous.items()},
            'total_population': self.total_population,
            'grid': self.grid,
        }

    @staticmethod
    def load(data, verbose):
        world = World(colours=data['colours'], verbose=verbose)
        world.by_name = {name: Municipality.load(mun)
                         for name, mun in data['by_name'].items()}

        world.municipalities = [mun for name, mun in world.by_name.items()]
        world.total_population = data['total_population']
        world.contiguous = {name: set(mlist)
                            for name, mlist in data['contiguous'].items()}
        world.grid = data['grid']
        world.set_grid()

        return world

    def get_contiguous_candidates(self, mun):
        x, y = mun.grid
        # Compose all possible grids (adjacent)
        grids = [(i, j) for i in range(x - 1, x + 2)
                 for j in range(y - 1, y + 2)]

        candidates = []
        # print 'mun', mun, 'for grid', mun.grid, 'testing grids: ', grids
        for grid in grids:
            candidates.extend(self.by_grid[grid])

        return candidates

    def prepare(self, verbose):
        self.municipalities = tuple(self.municipalities)
        self.by_name = {mun.name: mun for mun in self.municipalities}

        self.total_population = sum(mun.population
                                    for mun in self.municipalities)

        self.set_grid()

        # Check contiguous municipalities
        self.contiguous = defaultdict(set)
        self.boxes = defaultdict(set)

        if verbose:
            print('Preparing geometry')
            bar = progressbar.ProgressBar()
            iterator = bar(self.municipalities)
        else:
            iterator = self.municipalities

        for mun1 in iterator:
            for mun2 in self.municipalities:
            # for mun2 in self.get_contiguous_candidates(mun1):
                # if mun1.contiguous(mun2):
                if mun1.shape_contiguous(mun2):
                    self.contiguous[mun1.name].add(mun2.name)

        self.prepare_colours(verbose)

    def set_grid(self):
        # Assign shapelies
        self.bounds = Box(float('Inf'), float('Inf'), 0.0, 0.0)
        for mun in self.municipalities:
            left, bottom, right, top = mun.bounds
            new_left = min(self.bounds.left, left)
            new_bottom = min(self.bounds.bottom, bottom)
            new_right = max(self.bounds.right, right)
            new_top = max(self.bounds.top, top)
            self.bounds = Box(left=new_left, bottom=new_bottom,
                              right=new_right, top=new_top)
        self.grid = self.create_grid(self.bounds)

        self.by_grid = defaultdict(list)
        for mun in self.municipalities:
            coord = mun.grid_coordinate(self.grid)
            self.by_grid[coord].append(mun)

    def prepare_colours(self, verbose):
        # Assign colours
        bar = progressbar.ProgressBar()
        if verbose:
            print('Assigning colours')
            bar = progressbar.ProgressBar()
            iterator = bar(self.municipalities)
        else:
            iterator = self.municipalities

        last_result_colour = list(self.colours)[-1]
        for mun in iterator:
            available_colours = set(self.colours)
            for cont in self.contiguous[mun.name]:
                mun_cont = self.by_name[cont]
                if mun_cont.colour and mun_cont.colour in available_colours:
                    available_colours.remove(mun_cont.colour)

            if available_colours:
                mun.colour = choice(list(available_colours))
            else:
                mun.colour = last_result_colour

    def release_from_region(self, municipality):
        '''
        Release form current region and free
        '''
        # Find which region is holding it
        for region in self.regions:
            if municipality in region.municipalities:
                region.remove(municipality)

    def interactive_regions(self, colours):
        '''
        Ask for the seeding regions interactively
        '''
        print('INTERACTIVE MODE')
        print('Please select municipalities to create regions '
              'taking them as base')
        self.regions = []

        colours = colours[:]

        while True:
            print(u'Regions: {}'.format(self.regions))
            print('"q", "quit" or "exit" to finish input')
            print('"Enter" to select default')
            by_size = sorted((m for m in self.municipalities
                              if m.name in self.in_no_region),
                             key=lambda s: s.population,
                             reverse=True)
            default = by_size[0]
            prompt = u'Please select a municipality [{name}]: '
            print prompt.format(name=default.name),
            name = raw_input()
            name = name.strip()
            if name in ('q', 'exit', 'quit'):
                break

            mun = default
            if name:
                mun = self.by_name.get(name.decode('utf-8'), None)

            if not mun:
                print('"{}" is not understood'.format(name))
                continue

            colour = colours.pop(0)
            new_region = Region(self, colour, verbose=self.verbose,
                                name=mun.name)
            new_region.add(mun)
            self.regions.append(new_region)

    def make_regions(self, num_regions, colours, interactive, seed):
        '''
        Generate regions that have the same population
        '''
        print('Total population: {}'.format(self.total_population))
        target_population = self.total_population / num_regions
        tmpl = 'Making {reg} regions with aprox {target} population each'
        print(tmpl.format(reg=num_regions, target=target_population))

        self.in_no_region = {m.name for m in self.municipalities}
        print('total municipalities: {}'.format(len(self.in_no_region)))

        if seed:
            self.regions = []
            for index, ((name, muns), colour) in enumerate(zip(seed, colours)):
                region = Region(self, colour, verbose=self.verbose, name=name)
                for mun in muns.split(u'£'):
                    municipality = self.by_name[mun.strip()]
                    region.add(municipality)
                self.regions.append(region)

        else:
            if interactive:
                self.interactive_regions(colours)
            else:
                self.regions = [Region(self, colour, verbose=self.verbose)
                                for colour in colours[:num_regions]]
                # Base each group on the biggest municipality
                by_size = sorted(self.municipalities,
                                 key=lambda s: s.population,
                                 reverse=True)

                print('Creating {} regions'.format(len(self.regions)))
                for region, biggest in zip(self.regions, by_size):
                    region.add(biggest)

        print('Regions', self.regions)

        print('Adding municipalities to regions')
        assigning_regions = self.regions[:]

        ALGORITHM = 2
        if ALGORITHM == 1:
            print('Using algorithm 1')
            self.make_regions_1(assigning_regions)
        elif ALGORITHM == 2:
            print('Using algorithm 2')
            self.make_regions_2(assigning_regions)
        else:
            raise Exception('Region algorithm error')

        self.final_assignments()

        # Sort them to print them by size
        self.regions.sort(key=lambda x: x.population, reverse=True)

    def make_regions_1(self, assigning_regions):

        print 'Regions', assigning_regions

        # Add one municipality at a time
        bar = progressbar.ProgressBar()
        for i in bar(xrange(2 * len(self.in_no_region))):
            # Get the region with the least population
            # If any region is about to be closed, will always be selected
            def sorting(region):
                # Check available contiguous
                cnames = {c.name for c in region.contiguous}
                available = cnames & self.in_no_region
                if len(available) < 5:
                    return 5 - len(available)
                return region.population

            assigning_regions.sort(key=sorting)

            if not assigning_regions:
                break

            region = assigning_regions[0]

            if not region.available_contiguous(self.in_no_region):
                print 'removing region', region
                assigning_regions.pop(0)
                continue

            # Add one of the the biggest municipality that is contiguous
            # And that is available
            candidates = sorted(region.contiguous,
                                key=lambda x: x.population, reverse=True)
            for candidate in candidates:
                if region.add(candidate):
                    break

        if not self.in_no_region:
            return

        counter = 1000
        # Assing the remaining municipalities, which are small
        print('Some regions are locked, assigning rest of municipalities')
        bar = progressbar.ProgressBar()
        for i in bar(xrange(counter)):
            if not self.in_no_region:
                break
            added = False
            candidates = {self.by_name[c] for c in self.in_no_region}

            # Get the region with the least population
            self.regions.sort(key=lambda x: x.population)
            for region in self.regions:
                region_candidates = candidates & region.contiguous
                if region_candidates:
                    candidate = sorted(region_candidates,
                                       key=lambda x: x.population,
                                       reverse=True)
                for candidate in candidates:
                    if region.add(candidate, check_contiguous=True):
                        added = True
                        break
                if added:
                    break

            # else:
            #     # The region is bloqued to obtain another municipality
            #     # Force it out grabbing the biggest closest Municipality
            #     # that shares the most boundaries
            #     to_force = sorted([(c, self.border_region(region, c))
            #                        for c in candidates],
            #                       key=lambda x: x[1], reverse=True)
            #     print('1', to_force)

            #     def order_by_border(candidate):
            #         border = self.border_region(region, candidate)
            #         population = candidate.population
            #         return (border, population)

            #     to_force = sorted(candidates, key=order_by_border,
            #                       reverse=True)

            #     print('2', to_force)
            #     forced_candidate = candidates[0]
            #     forced_candidate = to_force[0]
            #     self.release_from_region(forced_candidate)
            #     region.add(forced_candidate)

    def final_assignments(self):
        ''' Assign exclaves or islands '''

        if self.in_no_region:
            print('Some municipalities are not assigned, probably '
                  'exclaves or islands')
            print('Assigning them to the closest region')
            if self.verbose:
                print("To be assigned", self.in_no_region)

            bar = progressbar.ProgressBar()
            for mun_name in bar(list(self.in_no_region)):
                mun = self.by_name[mun_name]
                # Calculate all distances
                # TODO, this may be slow
                regions = ((region, region.shape.distance(mun.shapely))
                           for region in self.regions)
                min_region, _ = min(regions, key=lambda x: x[1])
                min_region.add(mun)

        if self.in_no_region:
            print("Still somewhere is not assigned", self.in_no_region)

    def make_regions_2(self, assigning_regions):

        iterations = len(self.in_no_region) + len(assigning_regions)

        bar = progressbar.ProgressBar()
        for i in bar(xrange(iterations)):
            if not assigning_regions:
                break
            # Get the region with the least population
            assigning_regions.sort(key=lambda x: x.population)
            region = assigning_regions[0]

            # Add one of the the biggest municipality that is contiguous
            # And has the closest centre to the region
            region_centre = region.shape_centre

            def centre_distance(mun):
                if not region_centre:
                    return (0, -mun.population)

                distance = region_centre.distance(mun.shapely)
                # Split distance in 50 km blocks
                return (int(distance / 50), -mun.population)

            candidates = sorted((m for m in region.contiguous if m.name in self.in_no_region),
                                key=centre_distance, reverse=False)
            for candidate in candidates:
                if region.add(candidate, update_shape=True):
                    break

            else:
                # Region is locked, remove
                print 'region {} is locked, remove'.format(region)
                assigning_regions.pop(0)

        print self.in_no_region
        print 'Still unassigned', len(self.in_no_region)

    def print_regions(self):
        if self.regions is None:
            print('No groups so far')

        print('There are {} regions'.format(len(self.regions)))

        # Get the percentage difference from the ideal
        populations = [region.population for region in self.regions]
        ideal = self.total_population / len(populations)
        min_pop = 1 - (1. * min(populations)) / ideal
        max_pop = (1. * max(populations)) / ideal - 1

        print(('The regions are within -{:.2%}% / +{:.2%}% '
               'from ideal').format(min_pop, max_pop))

        for region in self.regions:
            region.print_region()

    def plot_regions(self):
        for region in self.regions:
            region.plot()

        l_colour = [Patch(color=region.colour) for region in self.regions]
        tmpl = u'ER "{0.name}" ({0.human_size})'
        l_text = [tmpl.format(region) for region in self.regions]
        pyplot.legend(l_colour, l_text)

    def population_histogram(self, num_groups, same_size=True):
        '''
        Return a list with municipalites grouped by population.
        Generate different num_groups, but all should have aprox
        the same number of municipalities

        This format:

        [
            {
                'min_population':
                'max_population':
                'number':
                'municipalities': (mun1, mun2...)
            }
            ...
        ]
        '''
        # Obtain the maximum and minumum population
        all_populations = sorted(mun.population for mun in self.municipalities)
        max_population = all_populations[-1]
        min_population = all_populations[0]
        EQUAL_GROUPS = False
        if EQUAL_GROUPS:
            group_size = len(all_populations) / num_groups
            cutting_points = [all_populations[i * group_size]
                              for i in xrange(1, num_groups)]
        else:
            step = (max_population - min_population) / num_groups
            cutting_points = [step * i for i in xrange(num_groups)]

        if self.verbose:
            print(cutting_points)

        histogram = []
        candidates = sorted(self.municipalities[:], key=lambda x: x.population)
        for cutting_point in reversed(cutting_points):
            municipalities = []
            while candidates[-1].population >= cutting_point:
                candidate = candidates.pop(-1)
                municipalities.insert(0, candidate)
                if not candidates:
                    break

            if municipalities:
                max_pop = max(m.population for m in municipalities)
                min_pop = min(m.population for m in municipalities)
            else:
                max_pop = 0
                min_pop = 0

            result = {
                'municipalities': municipalities,
                'number': len(municipalities),
                'max_population': max_pop,
                'min_population': min_pop,
            }
            histogram.append(result)

        # Last group
        if candidates:
            result = {
                'municipalities': candidates,
                'number': len(candidates),
                'max_population': max(m.population for m in candidates),
                'min_population': min(m.population for m in candidates),
            }
            histogram.append(result)

        if self.verbose:
            print(histogram)
        return histogram
