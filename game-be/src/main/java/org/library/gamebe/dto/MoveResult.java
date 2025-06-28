package org.library.gamebe.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.library.gamebe.util.MoveStatus;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class MoveResult {
    private MoveStatus status;
    private String message;
}