#(Newton's mthod with finite differences // derivative of function)
#lambert's solution --> function of time of flight
#scalar valued, single variable function, secant function. 
#repeat the same projectory in getting back to earth.

def calc_vinfinity( tof, args ):

    r1_planent1 = spice.spkgps( args[ 'planet1_ID' ], 
        args[ 'et0' ] + tof, args[ 'frame' ], args[ 'center_ID'] )[ 0 ]

    v0_sc_depart, v1_sc_arrive = lt.lamberts_universal_variables(
        args[ 'state0_planet0' ][ :3 ], r1_planet1, tof, {
             'mu' : args[ 'mu' ], 'tm': args[ 'tm' ]
        }
    )

    vinf = nt.norm( v0_sc_depart - args[ 'state0_planet0' ][ 3: ] )
    return args[ 'vinf' ] - vinf 