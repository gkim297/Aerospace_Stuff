def vinfinity_match( planet0, planet1, v0_sc, et0, tof0, args = {} ):
   
    #given an incoming v-infinity vector to planet0, calculate outgoing v-infinity vector
    #that will arrive at planet 1 after time of flight (tof) where incoming & outgoing
    #v-infinity vectors at planet0 have equal magnitute. 

    _args = {
        'et0'                       : et0,
        'planet1_ID'                : planet1,
        'frame'                     : 'ECLIPJ2000',
        'center_ID'                 : 0,
        'mu'                        : pd.sun[ 'mu' ],
        'tm'                        : 1,
        'diff_step'                 : 1e-3,
        'tol'                       : 1e-4
    }

    for key in args.keys(): 
        _args[ key ] = args[ key ]

    _args[ 'state0_planet0' ] = spice.spkgeo( planet0, et0,
        _args[ 'frame' ], args[ 'center_ID' ])[ 0 ]

    _args[ 'vinf' ] = nt.norm( v0_sc - _args [ 'state0_planet0'][ 3: ] )

    tof, steps = nt.newton_root_single_fd(
        calc_vinfinity, tof0, _args
    )

#earth to venus --> lambert's problems solver, to mars next, calculate equal incoming & outgoing v-infinity magnitutes.