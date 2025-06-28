package org.library.gamebe.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;
import org.library.gamebe.util.Color;
import org.library.gamebe.util.PieceType;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Piece {
    @JsonProperty("type")
    private PieceType type;

    @JsonProperty("color")
    private Color color;
}