package org.library.usersbe.repository;

import jakarta.transaction.Transactional;
import org.library.usersbe.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.UUID;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    User getUserByUsername(String username);

    void delete(User user);

    User findByUuid(UUID uuid);

    @Modifying
    @Transactional
    @Query("UPDATE User u SET u.username = :#{#user.username}, u.password = :#{#user.password}, " +
            "u.firstName = :#{#user.firstName}, u.lastName = :#{#user.lastName}, " +
            "u.email = :#{#user.email}, u.phone = :#{#user.phone}, u.role = :#{#user.role} " +
            "WHERE u.uuid = :uuid")
    void updateUserByUuid(UUID uuid, User user);

    boolean existsUserByUuid(UUID uuid);
}
