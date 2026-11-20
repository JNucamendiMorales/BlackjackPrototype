% ============================================================
% BLACKJACK PROTOTYPE
% SISTEMA EXPERTO - ESTRATEGIA BASICA
%
% Reglas:
% - Una baraja
% - Dealer se planta en 17 (S17)
% - Sin doblar
% - Sin dividir
% - Sin rendicion
%
% Acciones: pedir / plantarse
% ============================================================


% ------------------------------------------------------------
% VALOR DE LAS CARTAS
% ------------------------------------------------------------

valor_carta(a, 11).
valor_carta(j, 10).
valor_carta(q, 10).
valor_carta(k, 10).

valor_carta(2, 2).
valor_carta(3, 3).
valor_carta(4, 4).
valor_carta(5, 5).
valor_carta(6, 6).
valor_carta(7, 7).
valor_carta(8, 8).
valor_carta(9, 9).
valor_carta(10, 10).


% ------------------------------------------------------------
% CALCULO DE PUNTUACION
% ------------------------------------------------------------

% Suma inicialmente todos los ases como 11.

sumar_cartas([], 0, 0).

sumar_cartas([Carta|Resto], Total, Ases) :-
    valor_carta(Carta, Valor),
    sumar_cartas(Resto, Parcial, AsesParciales),
    Total is Parcial + Valor,
    (Carta == a ->
        Ases is AsesParciales + 1
    ;
        Ases = AsesParciales
    ).


% Ajusta los ases de 11 a 1 cuando es necesario.

ajustar_ases(Total, Ases, Ajustado) :-
    Total > 21,
    Ases > 0,
    !,
    NuevoTotal is Total - 10,
    NuevosAses is Ases - 1,
    ajustar_ases(NuevoTotal, NuevosAses, Ajustado).

ajustar_ases(Total, _, Total).


% Calcula la puntuacion final.

puntuacion(Cartas, Total) :-
    sumar_cartas(Cartas, Suma, Ases),
    ajustar_ases(Suma, Ases, Total).


% ------------------------------------------------------------
% DETECTAR MANO BLANDA
% ------------------------------------------------------------

% Una mano es blanda cuando conserva un as valorado en 11.

mano_blanda(Cartas) :-
    sumar_cartas(Cartas, Suma, Ases),
    Ases > 0,
    puntuacion(Cartas, Total),
    Total =< 21,
    Suma - Total < 10 * Ases.


% ------------------------------------------------------------
% ESTRATEGIA BASICA
% ------------------------------------------------------------

% Con 21 o mas puntos no se pide.

estrategia(Cartas, _, plantarse) :-
    puntuacion(Cartas, Total),
    Total >= 21,
    !.


% MANOS BLANDAS

% A,8 o A,9: plantarse.

estrategia(Cartas, _, plantarse) :-
    mano_blanda(Cartas),
    puntuacion(Cartas, Total),
    Total >= 19,
    !.


% A,7: plantarse contra dealer 2-8.
% Pedir contra 9, 10 o As.

estrategia(Cartas, Dealer, Accion) :-
    mano_blanda(Cartas),
    puntuacion(Cartas, 18),
    !,
    (
        Dealer >= 2,
        Dealer =< 8
    ->
        Accion = plantarse
    ;
        Accion = pedir
    ).


% Manos blandas de 17 o menos: pedir.

estrategia(Cartas, _, pedir) :-
    mano_blanda(Cartas),
    puntuacion(Cartas, Total),
    Total =< 17,
    !.


% MANOS DURAS

% 17 o mas: plantarse.

estrategia(Cartas, _, plantarse) :-
    puntuacion(Cartas, Total),
    Total >= 17,
    !.


% 13-16: plantarse contra dealer 2-6.

estrategia(Cartas, Dealer, plantarse) :-
    puntuacion(Cartas, Total),
    Total >= 13,
    Total =< 16,
    Dealer >= 2,
    Dealer =< 6,
    !.


% 12: plantarse contra dealer 4-6.

estrategia(Cartas, Dealer, plantarse) :-
    puntuacion(Cartas, 12),
    Dealer >= 4,
    Dealer =< 6,
    !.


% En los demas casos: pedir.

estrategia(_, _, pedir).
