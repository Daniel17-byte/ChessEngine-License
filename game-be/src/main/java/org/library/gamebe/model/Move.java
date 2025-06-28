package org.library.gamebe.model;

import lombok.*;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Move {
    private Position from;
    private Position to;
}